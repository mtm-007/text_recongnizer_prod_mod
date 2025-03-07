"""Test for Character Predictor Class."""
import os
from pathlib import Path
import unittest

from text_recognizer.character_predictor import CharacterPredictor

SUPPORT_DIRNAME = Path(__file__).parents[0].resolve() / 'support' / 'emnist'

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestCharacterPredictor(unittest.TestCase):
    def test_filename(self):
        predictor = CharacterPredictor()

        for filname in SUPPORT_DIRNAME.glob('*.png'):
            pred, conf = predictor.predict(str(filname))
            print(f'Prediction: {pred} at confidence: {conf} for image with character {filname.stem}')
            self.assertEqual(pred, filname.stem)
            self.assertGreater(conf, 0.65)
