# [VLA-0: Building State-of-the-Art VLAs with Zero Modification](vla0.github.io)

**Ankit Goyal, Hugo Hadfield, Xuning Yang, Valts Bulkis, Fabio Ramos**
NVIDIA
---

### Using VLA-0 in Your Research?

We'd love to hear about your work! If you've used VLA-0 in your research or projects, please reach out to [Ankit Goyal](mailto:ankgoyal@umich.edu) — we'd be happy to feature your work here.

---

## Installation

This section provides streamlined installation steps for training and evaluating VLA-0 (based on Qwen2.5-VL-3B) with LIBERO support.

### Installation Steps

**Step 0: Clone repository and submodules recursively**
```bash
git clone --recurse-submodules git@github.com:NVlabs/vla0.git
cd vla0
```

**Step 1: Create conda environment**
```bash
conda create -y -n vla0 python=3.10
conda activate vla0
```

**Step 2: Install package with qwen and libero extras**
```bash
PIP_REQ_EXTRAS=qwen,libero pip install --no-build-isolation -e ".[qwen,libero]"
```

**Step 3: Install dataset library with lerobot extras**
*Note: The "RoboVerse" library here is distinct from the RoboVerse paper. It is our standalone library for initializing various robot datasets. Currently tested for LeRobot datasets (version 0.1).*
```bash
cd libs/RoboVerse
PIP_REQ_EXTRAS=lerobot pip install --no-build-isolation -e ".[lerobot]"
cd ../..
```

### Evaluating on Libero
Download the trained model from [here](https://huggingface.co/ankgoyal/vla0-libero/tree/main) and place it under `vla0/runs`. The command generates videos of the runs and saves them in the run folder.
```bash
python eval/eval_libero.py --model_path ./runs/vla0/model_last.pth --task_suite_name libero_goal --task_name put_the_wine_bottle_on_top_of_the_cabinet --action_horizon 1 --ensemble_prediction 8 --task_id_count 10 --task_id_index 0
```

Notes:
- `--task_suite_name` and `--task_name` should be provided. For complete list run this bash command:
```python
python << EOF
from roboverse.evals.libero.eval import get_evaluation_tasks
print(get_evaluation_tasks())
EOF
```
- The above command makes a new prediction after each time step (`--action_horizon 1`) and ensembles 8 predictions (`--ensemble_prediction 8`)
- `--task_id_count` and `--task_id_index` can be used to parallelize multiple evaluations at the same time for the same task

To parse the results, use the following command with a `<run_id>` like `runs/vla0`. It uses the videos saved in the run folder to calculate the success rate. For the pre-trained model, we have provided the videos of each episode.
```bash
python logs/parse_libero_results.py <run_id>
```

### Training on Libero
```bash
python -m rv_train.train --exp-config ./configs/vla0.yaml
```
### Training on Custom LeRobot Data
- Create a dataset config file like `libs/RoboVerse/roboverse/configs/img_libero_aug.yaml`. It specifies the dataset. All the configurations and their default values are provided in `libs/RoboVerse/roboverse/configs.py`. The dataset config file overwrites the defaults. Some keys of interest are:
  - `horizon`: how many future timesteps to predict
  - `LEROBOT.repo_id`: LeRobot repository ID
  - `LEROBOT.action_key`: name of the action key in the LeRobot repo
  - `LEROBOT.state_key`: name of the state key
  - `LEROBOT.le_cam_list`: list of lerobot camera names to be used as input.
  - `IMAGE`: various image augmentation parameters

- Create a training config file like `configs/vla0.yaml`. Specify the path of the dataset config file in `DATALOADER.ROBOVERSE.cfg_path`. Note that `DATALOADER.ROBOVERSE.cfg_opts` further overwrites the settings in the dataset config. For example, `IMAGE.crop_img:0.875` would overwrite whatever is specified in the `cfg_path` yaml file.

- The training config file specifies the training run. All the configurations and their default values are provided in `rv_train/configs.py`. Some keys of interest are:
  - `MODEL.QWEN.original_action_dim`: number of dimensions in the action space. Should be `7` for 7-DoF joint pose.
  - `MODEL.QWEN.num_bins_actions`: for discretizing actions between 0 to num_bins_actions

- Launch training run for that exp-config like this:
```bash
python -m rv_train.train --exp-config <path_to_exp_cfg>
```

### Real-world Deployment
After training the model, you can deploy it in real-world (like LeRobot) by integrating VLA-0 inference directly into the deployment script. We recommend this approach if you are comfortable with it.

Alternatively, you can deploy the model as an API server for querying during real-world evaluation. First, update the checkpoint path by either changing `DEFAULT_CHECKPOINT` in `rv_train/deploy/model_manager.py` or by setting the `ROBOVERSE_DEPLOY_CHECKPOINT` environment variable. `ROBOVERSE_DEPLOY_CHECKPOINT` overrides `DEFAULT_CHECKPOINT`. Then run the following to start the API server:
```bash
ROBOVERSE_DEPLOY_CHECKPOINT=./runs/vla0/model_last.pth python rv_train/deploy/service.py
```

The server will print the IP address and port (default: 10000) when started. You can view the API documentation at `http://<server_ip>:10000/docs`.

On the client side (controlling the robot), you can query the server to get actions. A sample script for querying the server is provided in `rv_train/deploy/sample_client.py`. It can be run as:
```bash
python rv_train/deploy/sample_client.py
```
Replace the `SERVER_URL` in the script with the IP address printed by the server (use `localhost` if the server and client are on the same machine). This script tests both the "all actions at once" (`/predict_base64`) and "streaming" (`/predict_base64_stream`) endpoints of the action generator.

---

## Abstract

Vision-Language-Action models (VLAs) hold immense promise for enabling generalist robot manipulation. However, the best way to build them remains an open question. Current approaches often add complexity, such as modifying the existing vocabulary of a Vision-Language Model (VLM) with action tokens or introducing special action heads. Curiously, the simplest strategy of representing actions directly as text has remained largely unexplored.

This work introduces **VLA-0** to investigate this idea. We find that VLA-0 is not only effective; it is surprisingly powerful. With the right design, VLA-0 outperforms more involved models. On LIBERO, a popular benchmark for evaluating VLAs, VLA-0 outperforms all existing methods trained on the same robotic data. Furthermore, without large-scale robotics-specific training, it outperforms methods trained on large-scale robotic data. These findings also translate to the real world, where VLA-0 outperforms SmolVLA, a VLA model pre-trained on large-scale real data.

---

## Key Results

- :white_check_mark: **Best performance** on LIBERO among models without large-scale pretraining (94.7% average success rate)
- :white_check_mark: **Outperforms** methods with large-scale pretraining (π₀, π₀.₅-KI, GR00T-N1, MolmoAct)
- :white_check_mark: **Superior real-world performance** (+12.5% over SmolVLA on SO-100 robot)
- :white_check_mark: **No architectural changes** to the base VLM required

---

## Resources

- **Paper:** [PDF](https://vla0.github.io/data/root.pdf)
- **Website:** [https://vla0.github.io/](https://vla0.github.io/)

---

## Future Improvements

We welcome community contributions! Some areas we have identified are:

- **TensorRT-LLM Integration**: Our initial experiments suggest inference speed could be improved from 4 Hz to 6 Hz using optimized inference engines like TensorRT-LLM.
- **Lower Precision Deployment**: Implementing quantization and lower precision inference (e.g., INT8) could provide significant speed improvements with minimal accuracy loss.
- **LeRobot Version Compatibility**: Testing and ensuring compatibility with newer versions of LeRobot. We expect this to work with minimal changes but haven't validated it yet.
- **Direct LeRobot Integrations**: Make integrating with lerobot simpler and easier.

---

## Citation

If you find VLA-0 useful in your research, please consider citing:

```bibtex
@article{goyal2025vla0,
  title={VLA-0: Building State-of-the-Art VLAs with Zero Modification},
  author={Goyal, Ankit and Hadfield, Hugo and Yang, Xuning and Blukis, Valts and Ramos, Fabio},
  journal={arXiv preprint arXiv:2510.13054},
  year={2025}
}
```

---

## License

VLA-0 code and VLA-0 for Libero model are released under the [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/deed.en).

**Additional Information:**
- Built with Qwen2.5-VL-3B-Instruct
- Subject to [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) for the base model

---

## :e-mail: Contact

**For questions, please contact:**  
Ankit Goyal - [ankgoyal@umich.edu](mailto:ankgoyal@umich.edu)