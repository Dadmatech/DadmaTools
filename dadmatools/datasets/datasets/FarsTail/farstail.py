import json
import os
import io
import csv
from dadmatools.datasets.base import BaseDataset, DatasetInfo, BaseIterator
from dadmatools.datasets.dataset_utils import download_dataset, unzip_dataset, is_exist_dataset, DEFAULT_CACHE_DIR
URLS = ['https://raw.githubusercontent.com/dml-qom/FarsTail/master/data/Test-word.csv',
        'https://raw.githubusercontent.com/dml-qom/FarsTail/master/data/Train-word.csv',
        'https://raw.githubusercontent.com/dml-qom/FarsTail/master/data/Val-word.csv'
        ]
DATASET_NAME = "FarsTail"

def FarsTail(dest_dir=DEFAULT_CACHE_DIR):
    base_addr = os.path.dirname(__file__)
    info_addr = os.path.join(base_addr, 'info.py')
    DATASET_INFO = json.load(open(info_addr))
    dest_dir = os.path.join(dest_dir, DATASET_NAME)

    def get_FarsTail_item(dir_addr, fname):
        f_addr = os.path.join(dir_addr, fname)
        with io.open(f_addr, encoding="utf8") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                #skip headers
                if i == 0:
                    continue
                item = row[0].split('\t')
                if fname == 'Test-word.csv':
                    if len(item) != 5:
                        continue
                    yield {"premise": item[0], "hypothesis": item[1], "label": item[2], "hard(hypothesis)": item[3],
                           "hard(overlap)": item[4]}
                else:
                    if len(item) != 3:
                        continue
                    yield {"premise": item[0], "hypothesis": item[1], "label": item[2]}

    if not is_exist_dataset(DATASET_INFO, dest_dir):
        for url in URLS:
            download_dataset(url, dest_dir)
    info = DatasetInfo(info_addr=info_addr)
    train_iterator = get_FarsTail_item(dest_dir, 'Train-word.csv')
    test_iterator = get_FarsTail_item(dest_dir, 'Test-word.csv')
    val_iterator = get_FarsTail_item(dest_dir, 'Val-word.csv')
    sizes = DATASET_INFO['size']
    train_dataset = BaseIterator(train_iterator, num_lines=sizes['train'])
    test_dataset = BaseIterator(test_iterator, num_lines=sizes['test'])
    val_dataset = BaseIterator(val_iterator, num_lines=sizes['val'])
    iterators = {'train': train_dataset, 'test': test_dataset, 'val': val_dataset}
    dataset = BaseDataset(info=info)
    dataset.set_iterators(iterators)
    return dataset

