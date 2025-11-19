# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

# parse the results of the libero eval
# Usage: python logs/parse_libero_results.py <run_folder>

# We look for folder names *_eval_libero/ in the run folder
# In each of these folders, we look libero suites folders names libero_spatial, libero_object, libero_goal, libero_10
# In each libero suite folder, there are task folders named after the task name
# In each task folder, there are mp4 files. Each mp4 file is a video of the robot executing the task for one episode.
# We parse the mp4 files to calculate the success rate. The mp4 files are named as run<i>__success__<task_name>.mp4 and run<i>__failure__<task_name>.mp4

# For each *_eval_libero folder, we parse all the mp4 files and then provide the average success rate. Note results for every *_eval_libero folder are independent.
# 1. Each task
# 2. Each task suite, i.e. average success rate of all tasks in the suite
# 3. Overall average success rate over each task suite

import argparse
import glob
import os
import re
import sys
from collections import defaultdict


def sort_model_names(model_names):
    """
    Orders eval names with model_N.pth_<...> (ascending N) first,
    others go last.
    """

    def sort_key(name):
        match = re.match(r"model_(\d+)\.pth", name)
        if match:
            # Order by integer value of N
            return (0, int(match.group(1)), name)
        # Others get sorted after, in alpha order
        return (1, float("inf"), name)

    return sorted(model_names, key=sort_key)


def print_as_table(
    all_results,
    suite_name_order=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
):
    """
    Prints results in a neatly aligned tabular format with dynamic column widths and an OVERALL column.
    The OVERALL column displays both the average across suites and the total success rate.
    """
    # Gather all suite names across all evaluations
    if suite_name_order is None:
        all_suite_names = set()
        for suites in all_results.values():
            all_suite_names.update(suites.keys())
        suite_names = sorted(all_suite_names)
    else:
        suite_names = suite_name_order

    header = ["Evaluation"] + suite_names + ["OVERALL"]
    rows = []

    # Create table rows
    ordered_model_names = sort_model_names(all_results.keys())
    for model_name in ordered_model_names:
        row = [model_name]
        suites = all_results[model_name]
        suite_averages = []
        for suite_name in suite_names:
            tasks = suites.get(suite_name, {})
            if tasks:
                suite_success_rates = [res["success_rate"] for res in tasks.values()]
                suite_avg = sum(suite_success_rates) / len(suite_success_rates)
                total_success = sum(res["success"] for res in tasks.values())
                total_attempts = sum(res["total"] for res in tasks.values())
                suite_averages.append(suite_avg)
                # Add checkmark if total_attempts == 500
                mark = " ✓" if total_attempts == 500 else ""
                cell = f"{suite_avg * 100:.1f} ({total_success}/{total_attempts}, {len(tasks)} tasks){mark}"
            else:
                cell = "N/A"
            row.append(cell)

        # Calculate overall values as per the snippet you gave
        if suite_averages:
            overall_avg = sum(suite_averages) / len(suite_averages)
            total_success_all = sum(
                sum(results["success"] for results in suites[suite_name].values())
                for suite_name in suites.keys()
            )
            total_attempts_all = sum(
                sum(results["total"] for results in suites[suite_name].values())
                for suite_name in suites.keys()
            )
            # overall_rate = (
            #     total_success_all / total_attempts_all if total_attempts_all else 0
            # )
            overall_cell = (
                f"{overall_avg * 100:.1f} ({total_success_all}/{total_attempts_all})"
            )
        else:
            overall_cell = "N/A"
        row.append(overall_cell)
        rows.append(row)

    # Calculate maximum width for each column
    cols = [list(x) for x in zip(*([header] + rows))]
    col_widths = [max(len(str(cell)) for cell in col) + 2 for col in cols]

    # Function to format and print a row
    def format_row(row):
        return " | ".join(
            str(cell).ljust(width) for cell, width in zip(row, col_widths)
        )

    # Print header and divider
    print(format_row(header))
    print("-" * (sum(col_widths) + 3 * (len(col_widths) - 1)))

    # Print each data row
    for row in rows:
        print(format_row(row))


def main():
    parser = argparse.ArgumentParser(
        description="Parse libero evaluation results and calculate success rates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python logs/parse_libero_results.py /path/to/run/folder
  python logs/parse_libero_results.py /path/to/run/folder --verbose
  python logs/parse_libero_results.py /path/to/run/folder --pattern "model_a*"
        """,
    )

    parser.add_argument(
        "run_folder", help="Path to the run folder containing *_eval_libero directories"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed task and suite breakdowns (default: only show overall results)",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        help="Only process evaluation folders whose names match this pattern (supports wildcards)",
    )

    args = parser.parse_args()
    run_folder = args.run_folder

    if not os.path.exists(run_folder):
        print(f"Error: Run folder '{run_folder}' does not exist")
        sys.exit(1)

    print(f"Parsing libero results from: {run_folder}")

    # Step 2: Find all *_eval_libero/ folders
    eval_folders = glob.glob(os.path.join(run_folder, "*_eval_libero"))

    # Filter by pattern if provided
    if args.pattern:
        import fnmatch

        filtered_folders = []
        for folder in eval_folders:
            folder_name = os.path.basename(folder)
            if fnmatch.fnmatch(folder_name, args.pattern):
                filtered_folders.append(folder)
        eval_folders = filtered_folders
        print(
            f"Filtered by pattern '{args.pattern}': {len(eval_folders)} folders match"
        )

    if not eval_folders:
        if args.pattern:
            print(f"No *_eval_libero folders found matching pattern '{args.pattern}'")
        else:
            print("No *_eval_libero folders found")
        return

    print(f"Found {len(eval_folders)} evaluation folders:")
    for folder in eval_folders:
        print(f"  - {os.path.basename(folder)}")

    # Define expected libero suites
    libero_suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

    # Find libero suite folders in each eval folder
    suite_folders = {}
    for eval_folder in eval_folders:
        eval_name = os.path.basename(eval_folder)
        suite_folders[eval_name] = {}

        for suite in libero_suites:
            suite_path = os.path.join(eval_folder, suite)
            if os.path.exists(suite_path):
                suite_folders[eval_name][suite] = suite_path
                if args.verbose:
                    print(f"  Found suite: {eval_name}/{suite}")

    # Step 3: Parse mp4 files from task folders
    all_results = defaultdict(
        lambda: defaultdict(dict)
    )  # eval_name -> suite -> task -> results

    for eval_name, suites in suite_folders.items():
        for suite_name, suite_path in suites.items():
            if args.verbose:
                print(f"\nProcessing suite: {eval_name}/{suite_name}")

            # Find all task folders (subdirectories) in the suite
            task_folders = [
                d
                for d in os.listdir(suite_path)
                if os.path.isdir(os.path.join(suite_path, d))
            ]

            if args.verbose:
                print(f"  Found {len(task_folders)} task folders")

            for task_name in task_folders:
                task_path = os.path.join(suite_path, task_name)

                # Count success and failure mp4 files
                # Pattern: run<i>__success__<task_name>.mp4 and run<i>__failure__<task_name>.mp4
                success_pattern = os.path.join(task_path, "run*__success__*.mp4")
                failure_pattern = os.path.join(task_path, "run*__failure__*.mp4")

                success_files = glob.glob(success_pattern)
                failure_files = glob.glob(failure_pattern)

                success = len(success_files)
                failure = len(failure_files)
                total = success + failure

                if total > 0:
                    success_rate = success / total if total > 0 else 0
                    all_results[eval_name][suite_name][task_name] = {
                        "success": success,
                        "failure": failure,
                        "total": total,
                        "success_rate": success_rate,
                    }
                    if args.verbose:
                        print(
                            f"    {task_name}: {success}/{total} ({success_rate:.3f})"
                        )
                else:
                    print(f"    Warning: No mp4 files found in {task_path}")

    # Step 4: Calculate and display summary statistics for each evaluation folder independently
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS (Each evaluation folder treated independently)")
    print("=" * 80)

    # Process each evaluation folder separately
    for eval_name in sorted(all_results.keys()):
        print(f"\n{'='*60}")  # noqa
        print(f"RESULTS FOR: {eval_name}")
        print(f"{'='*60}")  # noqa

        suites = all_results[eval_name]

        if not suites:
            print("No results found for this evaluation folder")
            continue

        # Suite-level results for this evaluation folder
        print(f"\n1. SUITE-LEVEL RESULTS for {eval_name}:")
        print("-" * 50)

        suite_averages = []
        for suite_name in sorted(suites.keys()):
            tasks = suites[suite_name]
            if tasks:
                suite_success_rates = [
                    results["success_rate"] for results in tasks.values()
                ]
                suite_avg = sum(suite_success_rates) / len(suite_success_rates)
                suite_averages.append(suite_avg)

                # Calculate total successes and attempts for this suite
                total_success = sum(results["success"] for results in tasks.values())
                total_attempts = sum(results["total"] for results in tasks.values())

                print(
                    f"{suite_name}: {suite_avg:.3f} ({total_success}/{total_attempts}, {len(tasks)} tasks)"
                )

        # Overall results for this evaluation folder
        section_number = "2" if args.verbose else "OVERALL"
        print(f"\n{section_number}. OVERALL RESULTS for {eval_name}:")
        print("-" * 50)

        if suite_averages:
            overall_avg = sum(suite_averages) / len(suite_averages)
            total_success_all = sum(
                sum(results["success"] for results in suites[suite_name].values())
                for suite_name in suites.keys()
            )
            total_attempts_all = sum(
                sum(results["total"] for results in suites[suite_name].values())
                for suite_name in suites.keys()
            )

            print(
                f"Overall average across {len(suite_averages)} suites: {overall_avg:.3f}"
            )
            print(
                f"Total success rate: {total_success_all}/{total_attempts_all} ({total_success_all/total_attempts_all:.3f})"  # noqa
            )
        else:
            print("No suite results found")

    print()
    print()
    print(f"Results for: {run_folder}")
    print_as_table(all_results)


if __name__ == "__main__":
    main()
