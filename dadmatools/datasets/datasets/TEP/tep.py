import json
import os

from dadmatools.datasets.base import BaseDataset, DatasetInfo, BaseIterator
from dadmatools.datasets.dataset_utils import download_dataset, unzip_dataset, is_exist_dataset, DEFAULT_CACHE_DIR

URL = 'http://opus.nlpl.eu/download.php?f=TEP/v1/moses/en-fa.txt.zip'
DATASET_NAME = "TEP"

def TEP(dest_dir=DEFAULT_CACHE_DIR):
    base_addr = os.path.dirname(__file__)
    info_addr = os.path.join(base_addr, 'info.py')
    DATASET_INFO = json.load(open(info_addr))
    dest_dir = os.path.join(dest_dir, DATASET_NAME)

    def get_tep_item(dir_addr):
        en_f_name = os.path.join(dir_addr, 'TEP.en-fa.en')
        fa_f_name = os.path.join(dir_addr,'TEP.en-fa.fa')
        en_f = open(os.path.join(en_f_name))
        fa_f = open(os.path.join(fa_f_name))
        for fa_line, en_line in zip(fa_f, en_f):
            yield {'farsi': fa_line, 'eng': en_line}

    if not is_exist_dataset(DATASET_INFO, dest_dir):
        downloaded_file = download_dataset(URL, dest_dir)
        dest_dir = unzip_dataset(downloaded_file, dest_dir)
    info = DatasetInfo(info_addr=info_addr)
    train_iterator = get_tep_item(dest_dir)
    tep_size = DATASET_INFO['size']
    tep_iterator = BaseIterator(train_iterator, num_lines=tep_size)
    train = BaseDataset(info)
    train.set_iterators(tep_iterator)
    return train

