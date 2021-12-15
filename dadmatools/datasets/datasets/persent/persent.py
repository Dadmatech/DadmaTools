import os

import pandas as pd
from datasets.base import BaseDataset, DatasetInfo
from datasets.dataset_utils import download_with_progress, unzip_archive

URL = 'https://www.gelbukh.com/resources/persent/v1/PerSent.xlsx'
DATASET_NAME = "PerSent"

def PerSentLexicon(root):
    root = os.path.join(root, DATASET_NAME)

    def get_persent_lexicon(addr):
        df = pd.read_excel(addr, 'Dataset')
        for index, row_cells in df.iterrows():
            yield {'word': row_cells[0], 'pos': row_cells[1], 'sentiment':row_cells[2]}

    base_addr = os.path.dirname(__file__)
    info_addr = os.path.join(base_addr, 'info.json')
    info = DatasetInfo(info_addr=info_addr)
    downloaded_file = download_with_progress(URL, root)
    iterator = get_persent_lexicon(downloaded_file)
    lexicon = BaseDataset(iterator, info)
    return lexicon
