"""Define CharacterModel class."""
from typing import Callable, Dict, Tuple

import numpy as np

from text_recognizer.models.base import Model
from text_recognizer.datasets.emnist_dataset import EmnistDataset
from text_recognizer.networks.mlp import mlp
from text_recognizer.networks.lenet import lenet


class CharacterModel(Model):
    def __init__(self,
                 dataset_cls: type = EmnistDataset,
                 network_fn: Callable = lenet, #changed from mlp to lenet
                 dataset_args: Dict = None,
                 network_args: Dict = None):
        """Define the default dataset and network values of this model."""
        super().__init__(dataset_cls, network_fn, dataset_args, network_args)

    def predict_on_image(self, image: np.ndarray )-> Tuple[str, float]:
        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)

        pred_raw = self.network.predict(np.expand_dims(image, 0), batch_size=1).flatten()
        ind = np.argmax(pred_raw)
        confidence_of_prediction = pred_raw[ind]
        predicted_character = self.data.mapping[ind]

        return predicted_character,confidence_of_prediction
