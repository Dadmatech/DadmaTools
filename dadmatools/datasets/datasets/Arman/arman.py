import glob
import json
import os
from dadmatools.datasets.base import DatasetInfo, BaseDataset, BaseIterator
from dadmatools.datasets.dataset_utils import is_exist_dataset, unzip_dataset, download_dataset, DEFAULT_CACHE_DIR

URL = 'https://raw.githubusercontent.com/HaniehP/PersianNER/master/ArmanPersoNERCorpus.zip'
DATASET_NAME = "ARMAN"


def ARMAN(dest_dir=DEFAULT_CACHE_DIR):
    base_addr = os.path.dirname(__file__)
    info_addr = os.path.join(base_addr, 'info.py')
    DATASET_INFO = json.load(open(info_addr))
    dest_dir = os.path.join(dest_dir, DATASET_NAME)

    def get_arman_item(dir_addr, pattern):
        pattern = os.path.join(dir_addr, pattern)
        for f_addr in glob.iglob(pattern):
            f = open(f_addr,encoding='utf-8')
            sentence = []
            for line in f:
                if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                    if len(sentence) > 0:
                        yield sentence
                        sentence = []
                    continue
                splits = {'token': line.split(' ')[0], 'tag': line.split(' ')[1].replace('\n', '')}
                sentence.append(splits)

            if len(sentence) > 0:
                yield sentence
                sentence = []

    if not is_exist_dataset(DATASET_INFO, dest_dir):
        downloaded_file = download_dataset(URL, dest_dir)
        dest_dir = unzip_dataset(downloaded_file, dest_dir)

    train_iterator = get_arman_item(dest_dir, 'train*')
    test_dataset = get_arman_item(dest_dir, 'test*')
    # dev_iterator = get_arman_item(dir_addr, 'dev')
    # train = BaseDataset(train_iterator)
    # test = BaseDataset(test_iterator)

    info = DatasetInfo(info_addr=info_addr)
    train_size = DATASET_INFO['size']['train']
    test_size = DATASET_INFO['size']['test']
    train_iterator = BaseIterator(train_iterator, num_lines=train_size)
    test_iterator = BaseIterator(test_dataset, num_lines=test_size)
    iterators = {'train': train_iterator, 'test': test_iterator}
    dataset = BaseDataset(info=info)
    dataset.set_iterators(iterators)
    return dataset
