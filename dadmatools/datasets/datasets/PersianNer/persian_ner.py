import glob
import json
import os
from dadmatools.datasets.base import BaseDataset, DatasetInfo, BaseIterator
from dadmatools.datasets.dataset_utils import download_dataset, is_exist_dataset, DEFAULT_CACHE_DIR


URLS = ['https://raw.githubusercontent.com/Text-Mining/Persian-NER/master/Persian-NER-part1.txt',
        'https://raw.githubusercontent.com/Text-Mining/Persian-NER/master/Persian-NER-part2.txt',
        'https://raw.githubusercontent.com/Text-Mining/Persian-NER/master/Persian-NER-part3.txt',
        'https://raw.githubusercontent.com/Text-Mining/Persian-NER/master/Persian-NER-part4.txt',
        'https://raw.githubusercontent.com/Text-Mining/Persian-NER/master/Persian-NER-part5.txt'
        ]
DATASET_NAME = "PersianNer"

def PersianNer(dest_dir=DEFAULT_CACHE_DIR):
    base_addr = os.path.dirname(__file__)
    info_addr = os.path.join(base_addr, 'info.py')
    DATASET_INFO = json.load(open(info_addr))
    dest_dir = os.path.join(dest_dir, DATASET_NAME)

    def get_PersianNer_item(dir_addr, pattern):
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
                splits = {'token': line.split('\t')[0], 'tag': line.split('\t')[1].replace('\n', '')}
                sentence.append(splits)

            if len(sentence) > 0:
                yield sentence
                sentence = []

    if not is_exist_dataset(DATASET_INFO, dest_dir):
        for url in URLS:
            download_dataset(url, dest_dir)
    info = DatasetInfo(info_addr=info_addr)
    size = DATASET_INFO['size']
    iterator = BaseIterator(get_PersianNer_item(dest_dir, 'Persian-NER-part*'), num_lines=size)
    dataset = BaseDataset(info)
    dataset.set_iterators(iterator)
    return dataset

