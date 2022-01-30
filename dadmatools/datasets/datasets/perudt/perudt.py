import json
import os
from dadmatools.datasets.base import BaseDataset, DatasetInfo, BaseIterator
from dadmatools.datasets.dataset_utils import download_dataset, is_exist_dataset, DEFAULT_CACHE_DIR
from conllu import parse_incr
URLS = ['https://github.com/UniversalDependencies/UD_Persian-PerDT/raw/master/fa_perdt-ud-train.conllu',
        'https://github.com/UniversalDependencies/UD_Persian-PerDT/raw/master/fa_perdt-ud-dev.conllu',
        'https://github.com/UniversalDependencies/UD_Persian-PerDT/raw/master/fa_perdt-ud-test.conllu'
        ]
DATASET_NAME = "PerUDT"

def PerUDT(dest_dir=DEFAULT_CACHE_DIR):
    base_addr = os.path.dirname(__file__)
    info_addr = os.path.join(base_addr, 'info.py')
    DATASET_INFO = json.load(open(info_addr))
    dest_dir = os.path.join(dest_dir, DATASET_NAME)

    def get_perudt_item(dir_addr, fname):
        f_addr = os.path.join(dir_addr, fname)
        data_file = open(f_addr, "r", encoding="utf-8")
        for tokenlist in parse_incr(data_file):
            yield tokenlist

    if not is_exist_dataset(DATASET_INFO, dest_dir):
        for url in URLS:
            download_dataset(url, dest_dir)
    info = DatasetInfo(info_addr=info_addr)
    train_iterator = get_perudt_item(dest_dir, 'fa_perdt-ud-train.conllu')
    test_iterator = get_perudt_item(dest_dir, 'fa_perdt-ud-test.conllu')
    dev_iterator = get_perudt_item(dest_dir, 'fa_perdt-ud-dev.conllu')
    sizes = DATASET_INFO['size']
    train_iterator = BaseIterator(train_iterator, num_lines=sizes['train'])
    test_iterator = BaseIterator(test_iterator,  num_lines=sizes['test'])
    dev_iterator = BaseIterator(dev_iterator,  num_lines=sizes['dev'])
    iterators = {'train': train_iterator, 'test': test_iterator, 'dev': dev_iterator}
    dataset = BaseDataset(info)
    dataset.set_iterators(iterators)
    return dataset