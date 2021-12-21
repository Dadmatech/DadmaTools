import json
import os
from dadmatools.datasets.base import BaseDataset, DatasetInfo
from dadmatools.datasets.dataset_utils import download_dataset, is_exist_dataset, DEFAULT_CACHE_DIR

URLS = ['https://github.com/UniversalDependencies/UD_Persian-PerDT/raw/master/fa_perdt-ud-train.conllu',
        'https://github.com/UniversalDependencies/UD_Persian-PerDT/raw/master/fa_perdt-ud-dev.conllu',
        'https://github.com/UniversalDependencies/UD_Persian-PerDT/raw/master/fa_perdt-ud-test.conllu'
        ]
DATASET_NAME = "PerUDT"

def UPDT(dest_dir=DEFAULT_CACHE_DIR):
    base_addr = os.path.dirname(__file__)
    info_addr = os.path.join(base_addr, 'info.json')
    DATASET_INFO = json.load(open(info_addr))
    dest_dir = os.path.join(dest_dir, DATASET_NAME)

    def get_perudt_item(dir_addr, fname):
        f_addr = os.path.join(dir_addr, fname)
        f = open(f_addr)
        sentence = []
        for line in f:
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence) > 0:
                    yield sentence
                    sentence = []
                continue
            splits = line.split(' ')
            sentence.append([splits[0], splits[-1].rstrip("\n")])

            if len(sentence) > 0:
                yield sentence

        if not is_exist_dataset(DATASET_INFO, dest_dir):
            for url in URLS:
                download_dataset(url, dest_dir)
        info = DatasetInfo(info_addr=info_addr)
        train_iterator = get_perudt_item(dest_dir, 'fa_perdt-ud-train.conllu')
        test_iterator = get_perudt_item(dest_dir, 'fa_perdt-ud-test.conllu')
        dev_iterator = get_perudt_item(dest_dir, 'fa_perdt-ud-dev.conllu')
        sizes = DATASET_INFO['size']
        train_dataset = BaseDataset(train_iterator, info, num_lines=sizes['train'])
        test_dataset = BaseDataset(test_iterator, info, num_lines=sizes['test'])
        dev_dataset = BaseDataset(dev_iterator, info, num_lines=sizes['val'])
        return {'train': train_dataset, 'test': test_dataset, 'dev': dev_dataset}