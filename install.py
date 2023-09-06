import launch
import os

# TODO: add pip dependency if need extra module only on extension

if not launch.is_installed("cryptocode"):
    launch.run_pip("install cryptocode", "requirements for EliaiLora")

if 'COLAB_GPU' in os.environ:
    command = "apt install inxi"
    os.popen(command)