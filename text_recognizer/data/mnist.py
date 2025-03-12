""" Mnist DataModule"""
import argparse

from torch.utils.data import random_split
from torchvision.datasets import MNIST as TorchMnist
from torchvision import transforms
import urllib

from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info

DOWNLOAD_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded"

# NOTE: temp fix until https://github.com/pytorch/vision/issues/1938 is resolved
#from six.moves import urllib  # pylint: disable=wrong-import-position, wrong-import-order

opener =  urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)


class MNIST(BaseDataModule):
    """
    MNIST DataModule.
    """
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.data_dir = DOWNLOAD_DATA_DIRNAME
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.dims = (1, 28, 28)
        self.output_dims = (1, )
        self.mapping = list(range(10))
    
    def prepare_data(self, *args, **kwargs) ->None:
        """Download train and test MNIST data from PyTorch canonical source."""
        TorchMnist(self.data_dir, train= True, download = True)
        TorchMnist(self.data_dir, train= True, download = True)
    
    def setup(self, stage=None)->None:
        """Split into train, val, test, and set dims."""
        mnist_full = TorchMnist(self.data_dir, train=True, transform =self.transform)
        self.data_train, self.data_val = random_split(mnist_full, [55000, 5000])
        self.data_test= TorchMnist(self.data_dir, train=False, transform = self.transform)


if __name__ == '__main__':
    load_and_print_info(MNIST)