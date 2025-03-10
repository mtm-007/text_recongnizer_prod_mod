"""Class for loading the IAM dataset, which encompasses both paragraphs and lines, with associated utilities."""
import os
from typing import Dict, List
import xml.etree.ElementTree as ElementTree
import zipfile

from boltons.cacheutils import cachedproperty
import toml

from text_recognizer.datasets.dataset import Dataset, _download_raw_dataset


RAW_DATA_DIRNAME = Dataset.data_dirname() /'raw' /'iam'
METADATA_FILENAME = RAW_DATA_DIRNAME / 'metadata.toml'
EXTRACTED_DATASET_DIRNAME = RAW_DATA_DIRNAME/ 'iamdb'

DOWNSAMPLE_FACTOR = 2
LINE_REGION_PADDING = 0

"""
manually add data/raw/iam/metadata.toml

url = 'https://s3-us-west-2.amazonaws.com/fsdl-public-assets/iam/iamdb.zip'
filename = 'iamdb.zip'
sha256 = 'f3c9e87a88a313e557c6d3548ed8a2a1af2dc3c4a678c5f3fc6f972ba4a50c55'

"""


class IamDataset(Dataset):
    """
    "The IAM Lines dataset, first published at the ICDAR 1999, contains forms of unconstrained handwritten text,
    which were scanned at a resolution of 300dpi and saved as PNG images with 256 gray levels.
    From http://www.fki.inf.unibe.ch/databases/iam-handwriting-database

    The data split we will use is
    IAM lines Large Writer Independent Text Line Recognition Task (lwitlrt): 9,862 text lines.
        The validation set has been merged into the train set.
        The train set has 7,101 lines from 326 writers.
        The test set has 1,861 lines from 128 writers.
        The text lines of all data sets are mutually exclusive, thus each writer has contributed to one set only.
    """
    def __init__(self):
        os.makedirs(RAW_DATA_DIRNAME, exist_ok=True)
        self.metadata =  toml.load(METADATA_FILENAME)

    def load_or_generate_data(self):
        if not self.xml_filenames:
            self._download_iam()

    @property
    def xml_filenames(self):
        return list((EXTRACTED_DATASET_DIRNAME / 'xml').glob('*.xml'))

    @property
    def from_filenames(self):
        return list((EXTRACTED_DATASET_DIRNAME / 'forms').glob('*.jpg'))

    def _download_iam(self):
        os.makedirs(RAW_DATA_DIRNAME, exist_ok=True)
        curdir = os.getcwd()
        os.chdir(RAW_DATA_DIRNAME)
        _download_raw_dataset(self.metadata)
        _extract_raw_dataset(self.metadata)
        os.chdir(curdir)

    @property
    def from_filenames_by_id(self):
        return {filename.stem: filename for filename in self.from_filenames}

    @cachedproperty
    def line_regions_by_id(self):
        """Return a dict from name of IAM form to a list of line texts in it."""
        return {
            filename.stem: _get_line_regions_from_xml_file(filename)
            for filename in self.xml_filenames
        }

    def __repr__(self):
        """Print info about the dataset."""
        return (
            'IAM Dataset \n'
            f'Num forms: {len(self.xml_filenames)}\n'
        )


def _extract_raw_dataset(metadata):
    print('Extracting IAM data')
    with zipfile.ZipFile(metadata['filename'], 'r') as zip_file:
        zip_file.extractall()

def _get_line_strings_from_xml_file(filename: str) ->List[str]:
    """Get the text content of each line. Note that we replace &quot; with "."""
    xml_root_element = ElementTree.parse(filename).getroot()
    xml_line_elements = xml_root_element.findall('handwritten-part/line')
    return [el.attrib['text'].replace('&quot;', '"') for el in xml_line_elements]

def _get_line_regions_from_xml_file(filename: str) ->List[str]:
    """Get the line region dict for each line."""
    xml_root_element = ElementTree.parse(filename).getroot()
    xml_line_elements = xml_root_element.findall('handwritten-part/line')
    return [_get_line_regions_from_xml_elements(el) for el in xml_line_elements]

def _get_line_regions_from_xml_elements(xml_line) ->Dict[str, int]:
    """
    line (xml element): has x, y, width, and height attributes
    """
    word_elements = xml_line.findall('word/cmp')
    x1s = [int(el.attrib['x']) for el in word_elements]
    y1s = [int(el.attrib['y']) for el in word_elements]
    x2s = [int(el.attrib['x']) + int(el.attrib['width']) for el in word_elements]
    y2s = [int(el.attrib['y']) + int(el.attrib['height']) for el in word_elements]
    return {
        'x1' : min(x1s) // DOWNSAMPLE_FACTOR - LINE_REGION_PADDING,
        'y1' : min(y1s) // DOWNSAMPLE_FACTOR - LINE_REGION_PADDING,
        'x2' : min(x2s) // DOWNSAMPLE_FACTOR + LINE_REGION_PADDING,
        'y2' : min(y2s) // DOWNSAMPLE_FACTOR + LINE_REGION_PADDING
    }


def main():
    dataset = IamDataset()
    dataset.load_or_generate_data()
    print(dataset)


if __name__ == '__main__':
    main()
