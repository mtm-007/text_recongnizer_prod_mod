## for Python path mismath:
bash
`
pipenv install --system --dev  
export PYTHONPATH=$(pwd):$PYTHONPATH

`
## tensorflow and Cuda libraries mismatch fix
chatgpt link here:
https://chatgpt.com/share/67ad5671-8d7c-800a-ab38-39d6216ff71e

dpkg -l | grep nvidia-driver
if nothing appears:

sudo apt update
sudo apt install nvidia-driver-535

export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.1

pip install --upgrade nvidia-cudnn-cu12
pip install --upgrade nvidia-tensorrt-cu12

pip uninstall tensorflow

pip install tensorflow==2.15

# to download and process the dataset run
bash
`
pipenv run python text_recognizer/datasets/emnist_dataset.py
`

## To run the experiment(Training) run:
bash
`
pipenv run python training/run_experiment.py --save '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp",  "train_args": {"batch_size": 256}}'


pipenv run python training/run_experiment.py '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "lenet",  "train_args": {"batch_size": 256}}'

`