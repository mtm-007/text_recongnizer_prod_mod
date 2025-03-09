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


pipenv run python training/run_experiment.py --save '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "lenet",  "train_args": {"batch_size": 256}}'

python training/run_experiment.py --save '{"train_args": {"epochs": 16}, "dataset": "EmnistLinesDataset", "model": "LineModelCtc", "network": "line_lstm_ctc"}'


#### changed the num hidden layer from 2 to 3 to match test run
python training/run_experiment.py --save '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp", "network_args": {"num_layers": 3}, "train_args": {"batch_size": 256, "epochs" : 20}, "experiment_group": "Sample Experiments 2"}'

python training/run_experiment.py --save '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp", "network_args": {"num_layers": 4}, "train_args": {"batch_size": 256}, "experiment_group": "Sample Experiments 2"}'


python training/run_experiment.py --save '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "lenet", "train_args": {"batch_size": 256}, "experiment_group": "Sample Experiments 2"}'

python training/run_experiment.py --save '{"dataset": "IAMLinesDataset", "model": "LineModelCtc", "network": "line_lstm_ctc"}'

### line predector on iam dataset, run iam_dataset to get the iam dataset
python training/run_experiment.py --save '{"dataset": "IAMLinesDataset", "model": "LineModelCtc", "network": "line_lstm_ctc"}'
#### for iam paragraphs dataset
python training/run_experiment.py --save '{"dataset": "IAMParagraphsDataset", "model": "LineDetectorModel", "network": "fcn"}'

`
