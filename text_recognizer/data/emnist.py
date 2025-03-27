"""
EMNIST dataset. Downloads from NIST website and saves as .npz file if not already present.
"""
from pathlib import Path 
from typing import Sequence
import json
import shutil
import os 
import zipfile

from torchvision import transforms
import h5py
import toml 
import numpy as np

from text_recognizer.data.base_data_module import _download_raw_dataset,BaseDataModule, load_and_print_info
from text_recognizer.data.util import BaseDataset, split_dataset

NUM_SPECIAL_TOKENS = 4
SAMPLE_TO_BALANCE = True
TRAIN_FRAC = 0.8

RAW_DATA_DIRNAME = BaseDataModule.data_dirname() /"raw" / "emnist"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DL_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloads" / "emnist"
PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "emnist"
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / "byclass.h5"
ESSESNTIALS_FILENAME = Path(__file__).parents[0].resolve() / "emnist_essentials.json"


class EMNIST(BaseDataModule):
    """
    "The EMNIST dataset is a set of handwritten character digits derived from the NIST Special Database 19
    and converted to a 28x28 pixel image format and dataset structure that directly matches the MNIST dataset."
    From https://www.nist.gov/itl/iad/image-group/emnist-dataset

    The data split we will use is
    EMNIST ByClass: 814,255 characters. 62 unbalanced classes.
    """
    def __init__(self, args= None):
        super().__init__(args)

        if not os.path.exists(ESSESNTIALS_FILENAME):
            _download_raw_dataset()
        with open(ESSESNTIALS_FILENAME) as f:
            essentials = json.load(f)
        self.mapping = list(essentials["characters"])
        self.inverse_mapping = {v: k for k,v in enumerate(self.mapping)}
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dims = (1, *essentials["input_shape"]) # Extra dimension is added by ToTensor()
        self.output_dims = (1,)
    
    def prepare_data(self, *args,  **kwargs) -> None:
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            _download_raw_dataset()
        with open(ESSESNTIALS_FILENAME) as f:
            _essentials = json.load(f)
    
    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
                self.x_trainval = f["x_train"][:]
                self.y_trainval = f["y_train"][:].squeeze().astype(int)
            
            data_trainval = BaseDataset(self.x_trainval, self.y_trainval, transform = self.transform)
            self.data_train, self.data_val = split_dataset(base_dataset=data_trainval,fraction = TRAIN_FRAC, seed=42)

        if stage == "test" or stage is None:
            with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
                self.x_test = f["x_test"][:]
                self.y_test = f["y_test"][:].squeeze().astype(int)
            
            self.data_test = BaseDataset(self.x_test, self.y_test, transform = self.transform)
            
    def __repr__(self):
        basic = f"EMNIST Dataset\nNum classes : {len(self.mapping)}\nMapping: {self.mapping}\nDims: {self.dims}\n"
        if self.data_train is None and self.data_val is None and self.data_val is None:
            return basic
        
        x,y = next(iter(self.train_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}"
            f"Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Batch y stats: {(y.shape, y.stype, y.min(), y.max())}\n"
        )
        return basic + data
    
def _download_and_process_emnist():
    metadata= toml.load(METADATA_FILENAME)
    _download_raw_dataset(metadata, DL_DATA_DIRNAME)
    _process_raw_dataset(metadata["filename"], DL_DATA_DIRNAME)


def _process_raw_dataset(filename: str, dirname: path):
    print("Unzipping EMnist...")
    curdir = os.getcwd()
    os.chdir(dirname)
    zip_file = zipfile.ZipFile(filename, "r")
    zip_file.extract("matlab/emnist-byclass.mat")

    from scipy.io import loadmat

    print("Loading training data from .mat file")
    data = loadmat("matlab/emnist-byclass.mat")
    x_train = data["dataset"]["train"][0,0]["images"][0,0].reshape(-1, 28, 28).swapaxes(1,2)
    y_train = data["dataset"]["train"][0,0]["labels"][0,0] + NUM_SPECIAL_TOKENS
    x_test = data["dataset"]["test"][0,0]["images"][0,0].reshape(-1, 28, 28).swapaxes(1,2)
    y_test = data["dataset"]["test"][0,0]["labels"][0,0] + NUM_SPECIAL_TOKENS
    # NOTE that we add NUM_SPECIAL_TOKENS to targets, since these tokens are the first class indices

    if SAMPLE_TO_BALANCE:
        print("Balancing classes to reduce amount of data")
        x_train, y_train = _sample_to_balance(x_train, y_train)
        x_test, y_test = _sample_to_balance(x_test, y_test)
    
    print("Saving to hDFS in a compressed format")
    mapping = {int(k): chr(v) for k, v in data["dataset"]["mapping"][0,0]}
    characters = _augument_emnist_character(list(mapping.values()))
    essentials = {"characters" : characters, "input_shape": list(x_train.shape[1:])}
    with open(ESSENTIALS_FILANAME, "w") as f:
        json.dump(essentials, f)
    
    print("Cleaning up ...")
    shutil.rmtree("matlab")
    os.chdir(curdir)


def sample_to_balance(x, y):
    """Because the dataset is not balanced, we take at most the mean number of instances per class."""
    np.random.seed(42)
    num_to_sample = int(np.bincount(y.flatten()).mean())
    all_sampled_inds = []
    for label in np.unique(y.flatten()):
        inds = np.where(y == label)[0]
        sampled_inds = np.unique(np.random.choice(inds, num_to_sample))
        all_sampled_inds.append(sampled_inds)
    ind = np.concatenate(all_sampled_inds)
    x_sampled = x[inds]
    y_sampled = y[inds]
    return x_sampled, y_sampled


def _augument_emnist_characters(characters: Sequence[str]) -> Sequence[str]:
    """Augument the mapping with extra symbols."""
    # Extra characters from the IAM dataset
    iam_characters = [
        " ",
        "!",
        '"',
        "#",
        "&",
        "'",
        "(",
        ")",
        "*",
        "+",
        ",",
        "-",
        ".",
        "/",
        ":",
        ";",
        "?",
    ]
    return ["<B>", "<S>", "<E>", "<P>", *characters, *iam_characters]


if __name__ == "__main__":
    load_and_print_info()