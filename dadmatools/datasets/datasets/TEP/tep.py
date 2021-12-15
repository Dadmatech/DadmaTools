import os

from datasets.base import BaseDataset, SplittedDataset, DatasetInfo
from datasets.dataset_utils import download_with_progress, unzip_archive

URL = 'http://opus.nlpl.eu/download.php?f=TEP/v1/moses/en-fa.txt.zip'
DATASET_NAME = "TEP"

def TEP(root):
    root = os.path.join(root, DATASET_NAME)

    def get_tep_item(dir_addr):
        en_f_name = os.path.join(dir_addr, 'TEP.en-fa.en')
        fa_f_name = os.path.join(dir_addr,'TEP.en-fa.fa')
        en_f = open(os.path.join(en_f_name))
        fa_f = open(os.path.join(fa_f_name))
        for fa_line, en_line in zip(fa_f, en_f):
            yield {'farsi': fa_line, 'eng': en_line}


    downloaded_file = download_with_progress(URL, root)
    dir_addr = unzip_archive(downloaded_file, root)
    base_addr = os.path.dirname(__file__)
    info_addr = os.path.join(base_addr, 'info.json')
    info = DatasetInfo(info_addr=info_addr)
    train_iterator = get_tep_item(dir_addr)
    train = BaseDataset(train_iterator, info)
    return train

