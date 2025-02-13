"""CharacterPredictor class."""
from typing import Tuple, Union
import numpy as np

from text_recognizer.models import CharacterModel
import text_recognizer.util as util


class CharacterPredictor:
    """Given an image of a single hand written character, recognize it."""
    def __init__(self):
        self.model = CharacterModel()
        self.model.load_weights()

    def predict(self, image_or_file: Union[np.ndarray, str]) -> Tuple[str, float]:
        """Predict on a single image."""
        if isinstance(image_or_file, str):
            image = util.read_image(image_or_file, grayscale=True)
        else:
            image = image_or_file
        return self.model.predict_on_image(image)
    
    def evaluate(self, dataset):
        """Evaluate on dataset."""
        return self.model.evaluate(dataset.x_test, dataset.y_test)