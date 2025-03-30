from typing import Dict, Sequence
from pathlib import Path
from collections import defaultdict
import argparse

from torchvision import transforms
import h5py 
import numpy as np
import torch

from text_recognizer.data.util import BaseDataset
from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.emnist import EMNIST


DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "emnist_lines"
ESSENTIALS_FILENAME = Path(__file__).parent[0].resolve() / "emnist_lines_essentials.json"

MAX_LENGTH = 32
MIN_OVERLAP = 0
MAX_OVERLAP = 0.33
NUM_TRAIN = 10000
NUM_VAL = 2000
NUM_TEST = 2000


class EMNISTLines(BaseDataModule):
    """EMNIST Lines dataset: synthetic handwriting lines dataset made from EMNIST characters."""
    def __init__(    
        self,
        args: argparse.Namespace = None
    ):
        super().__init__(args)

        self.max_length = self.args.get("max_length", MAX_LENGTH)
        self.min_overlap = self.args.get("min_overlap", MIN_OVERLAP)
        self.max_overlap = self.args.get("max_overlap", MAX_OVERLAP)
        self.num_train = self.args.get("num_train", NUM_TRAIN)
        self.num_val = self.args.get("num_val", NUM_VAL)
        self.num_test = self.args.get("num_test", NUM_TEST)
        self.with_start_end_tokens = self.args.get("with_start_end_tokens", False)

        self.emnist = EMNIST()
        self.mapping = self.emnist.mapping
        self.dims = (
            self.emnist.dims[0],
            self.emnist.dims[1],
            self.emnist.dims[2] * self.max_length,
        )
        self.output_dims = (self.max_length, 1)
        self.transform = transforms.Compose([transforms.ToTensor()])
    

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--max_length", type=int, default = MAX_LENGTH, help = "Max length in characters")
        parser.add_argument("--min_overlap", type=float, default = MIN_OVERLAP, help = "Min overlap between characters in a line, between 0 and 1")
        parser.add_argument("--max_overlap", type=float, default = MAX_OVERLAP, help = "Max overlap between characters in a line, between 0 and 1")
        parser.add_argument("--with_start_end_tokens", action = "store_true", default = False)
        return parser


    @property
    def data_filename(self):
        return (
            DATA_DIRNAME
            / f"ml_{self.max_length}_o{self.min_overlap:f}_{self.max_overlap:f}_ntr{self.num_train}_ntv{self.num_val}_nte{self.num_test}_{self.with_start_end_tokens}.h5"
        )
    

    def prepare_data(self, *args, **kwargs)->None:
        if self.data_filename.exists():
            return
        np.random.seed(42)
        self._generate_data("train")
        self._generate_data("val")
        self._generate_data("test")
    
    def setup(self, stage: str = None) -> None:
        print("EMNISTLinesDataset loading data from HDF5...")
        if stage == "fit" or stage is None:
            with h5py.File(self.data_filename, "r") as f:
                x_train = f["x_train"][:]
                y_train = f["y_train"][:].astype(int)
                x_val = f["x_val"][:]
                y_val = f["y_val"][:].astype(int)

            self.data_train = BaseDataset(x_train, y_train, transform = self.transform)
            self.data_val = BaseDataset(x_val, y_val, transform = self.transform)
        
        if stage == "test" or stage is None:
            with h5py.File(self.data_filename, "r") as f:
                x_test = f["x_test"][:]
                y_test = f["y_test"][:].astype(int)
            self.data_test = BaseDataset(x_test, y_test, transform = self.transform)

    
    def __repr__(self)-> str:
        """Print info about dataset."""
        