# to download and process the dataset run
bash
`
pipenv run python text_recognizer/datasets/emnist_dataset.py
`

## To run the experiment(Training) run:
bash
`
pipenv run python training/run_experiment.py --save '{"dataset": "EmnistDataset", "model": "CharacterModel", "network": "mlp",  "train_args": {"batch_size": 256}}'

`