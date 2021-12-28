import glob
import json
import os
from dadmatools.datasets.base import BaseDataset, DatasetInfo, BaseIterator
from dadmatools.datasets.dataset_utils import download_dataset, is_exist_dataset, DEFAULT_CACHE_DIR, unzip_dataset

URL = 'https://drive.google.com/uc?id=1EC121uhkOFlsPAvsPMJ9TvBAwus_UkhN'
DATASET_NAME = "Peyma"

def Peyma(dest_dir=DEFAULT_CACHE_DIR):
    base_addr = os.path.dirname(__file__)
    info_addr = os.path.join(base_addr, 'info.py')
    DATASET_INFO = json.load(open(info_addr))
    dest_dir = os.path.join(dest_dir, DATASET_NAME)

    def get_peyma_item(dir_addr, pattern):
        pattern = os.path.join(dir_addr, pattern)
        for f_addr in glob.iglob(pattern):
            f = open(f_addr)
            sentence = []
            for line in f:
                if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                    if len(sentence) > 0:
                        yield sentence
                        sentence = []
                    continue
                line = line.replace('\n', '')
                token = {'token': line.split('\t')[0], 'tag':line.split('\t')[1]}
                sentence.append(token)

            if len(sentence) > 0:
                yield sentence

    if not is_exist_dataset(DATASET_INFO, dest_dir):
        downloaded_file = os.path.join(dest_dir, 'peyma.zip')
        download_dataset(URL, dest_dir, filename=downloaded_file)
        dest_dir = unzip_dataset(downloaded_file, dest_dir, zip_format='zip')
    info = DatasetInfo(info_addr=info_addr)
    iterator = get_peyma_item(dest_dir, 'peyma/*K/*')
    size = DATASET_INFO['size']
    data_iterator = BaseIterator(iterator, num_lines=size)
    dataset = BaseDataset(info)
    dataset.set_iterators(data_iterator)
    return dataset

