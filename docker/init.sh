# To be run inside the container from RV_train directory. It locally installs all the repositories we plan to edit.
# Usage:
# bash docker/init.sh

pip install -e .  # RV_train
pip install -e libs/RoboVerse  # RoboVerse
conda install ffmpeg -y  # required for lerobot
pip install tbparse
pip install peft==0.15.1 transformers==4.51.3 accelerate==1.6.0 qwen-vl-utils==0.0.10 # for copatibility with local training setup, this will be move to setup.py