"""Utility function for text_recognizer Module."""
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from typing import Union
from urllib.request import urlopen, urlretrieve
import base64
import hashlib
import os

import numpy as np
import cv2
from tqdm import tqdm


def read_image(image_url: Union[Path, str], grayscale=False) -> np.array:
    """Read image_url."""
    def read_image_from_filename(image_filename, imread_flag):
        return cv2.imread(str(image_filename), imread_flag)

    def read_image_from_url(image_url,imread_flag):
        url_response = urlopen(str(image_url))
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        return cv2.imdecode(img_array, imread_flag)

    imread_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    local_file = os.path.exists(image_url)
    try:
        img = None
        if local_file:
            img = read_image_from_filename(image_url, imread_flag)
        else:
            img = read_image_from_url(image_url, imread_flag)
        assert img is not None
    except Exception as e:
        raise ValueError("Could not load image from b64 {}: {}".format(image_url, e))
    return img


def read_b64_image(b64_string, grayscale=False):
    """Load base64-encoded images."""
    imread_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    try:
        _, b64_data = b64_string.split(',')
        return cv2.imdecode(np.frombuffer(base64.b64decode(b64_data), np.uint8), imread_flag)
    except Exception as e:
        raise ValueError("Could not load image from b64 {}: {}".format(b64_string, e))


def write_image(image: np.ndarray, filename: Union[Path, str]) -> None:
    cv2.imwrite(str(filename), image)

def compute_sha256(filename: Union[Path, str]):
    """Return SHA256 checksum of a file."""
    with open(filename, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


class TqdmUpTo(tqdm):
    """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py"""
    def update_to(self, blocks=1, bsize=1, tsize=None):
        """
        blocks : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(blocks* bsize - self.n) # will also set self.n = b * bsize

def download_url(url, filename):
    """Download a file from url to filename, with a progress bar"""
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1) as t:
        urlretrieve(url, filename, reporthook=t.update_to)
