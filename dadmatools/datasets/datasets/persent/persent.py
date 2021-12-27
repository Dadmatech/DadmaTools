import json
import os

import pandas as pd
from dadmatools.datasets.base import BaseDataset, DatasetInfo, BaseIterator
from dadmatools.datasets.dataset_utils import download_dataset, unzip_dataset, is_exist_dataset, DEFAULT_CACHE_DIR

URL = 'https://www.gelbukh.com/resources/persent/v1/PerSent.xlsx'
DATASET_NAME = "PerSent"

def PerSentLexicon(dest_dir=DEFAULT_CACHE_DIR):
    base_addr = os.path.dirname(__file__)
    info_addr = os.path.join(base_addr, 'info.py')
    DATASET_INFO = json.load(open(info_addr))
    dest_dir = os.path.join(dest_dir, DATASET_NAME)

    def get_persent_lexicon(addr):
        addr = os.path.join(addr, 'PerSent.xlsx')
        df = pd.read_excel(addr, 'Dataset')
        for index, row_cells in df.iterrows():
            yield {'word': row_cells[0], 'pos': row_cells[1], 'sentiment':row_cells[2]}

    if not is_exist_dataset(DATASET_INFO, dest_dir):
        downloaded_file = download_dataset(URL, dest_dir)
        dest_dir = unzip_dataset(downloaded_file, dest_dir)
    info = DatasetInfo(info_addr=info_addr)
    lexicon_size = DATASET_INFO['size']
    iterator = BaseIterator(get_persent_lexicon(dest_dir), num_lines=lexicon_size)
    dataset = BaseDataset(info)
    dataset.set_iterators(iterator)
    return dataset
