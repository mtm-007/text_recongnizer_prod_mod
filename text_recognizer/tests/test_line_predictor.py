"""Test for LinePredictor class."""
import os
from pathlib import Path
import unittest

import editdistance
import numpy as np

import text_recognizer.util as util
from text_recognizer.line_predictor import LinePredictor
from text_recognizer.datasets import IAMLinesDataset

SUPPORT_DIRNAME = Path(__file__).parents[0].resolve() / 'support'

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestEMnistLinesPredictor(unittest.TestCase):
    def test_filename(self):
        predictor = LinePredictor()

        for filename in (SUPPORT_DIRNAME / 'emnist_lines').glob('*.png'):
            pred, conf = predictor.predict(str(filename))
            true = str(filename.stem)
            edit_distance = editdistance.eval(pred, true) / len(pred)
            print(f'pred: "{pred}" | Confidence: {conf} | True: {true} | Edit distance: {edit_distance}')
            self.assertLess(edit_distance, 0.2)


class TestEmnistLinesPredictorVariableImageWidth(unittest.TestCase):
    def test_filename(self):
        predictor = LinePredictor()
        for filename in SUPPORT_DIRNAME.glob('*.png'):
            image = util.read_image(str(filename), grayscale=True)
            print('Saved image shape:', image.shape)
            image = image[:, :-np.random.randint(1, 150)]
            print('Randomly cropped image shape:', image.shape)
            pred, conf = predictor.predict(image)
            true = str(filename.stem)
            edit_distance = editdistance.eval(pred, true) / len(pred)
            print(f'pred: "{pred}" | Confidence: {conf} | True: {true} | Edit distance: {edit_distance}')
            self.assertLess(edit_distance, 0.2)


class TestIAMLinePredictor(unittest.TestCase):
    def test_filename(self):
        predictor = LinePredictor(IAMLinesDataset)

        for filename in (SUPPORT_DIRNAME / 'iam_lines').glob('*.png'):
            pred, conf = predictor.predict(str(filename))
            true = str(filename.stem)
            if pred:
                edit_distance = editdistance.eval(pred, true) / len(pred)
            else:
                edit_distance = 0
            print(f'Pred: "{pred}" | Confidence: {conf} | True: {true} | Edit distance: {edit_distance}')
