## for Python path mismath:
bash
`
export PYTHONPATH=$(pwd):$PYTHONPATH

`
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