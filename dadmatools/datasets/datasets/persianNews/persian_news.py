import csv
import io
import json
import os
from dadmatools.datasets.base import BaseDataset, DatasetInfo, BaseIterator
from dadmatools.datasets.dataset_utils import download_dataset, is_exist_dataset, DEFAULT_CACHE_DIR, unzip_dataset

URL = 'https://drive.google.com/uc?id=1B6xotfXCcW9xS1mYSBQos7OCg0ratzKC'
DATASET_NAME = "PersianNews"

def PersianNews(dest_dir=DEFAULT_CACHE_DIR):
    base_addr = os.path.dirname(__file__)
    info_addr = os.path.join(base_addr, 'info.py')
    DATASET_INFO = json.load(open(info_addr))
    dest_dir = os.path.join(dest_dir, DATASET_NAME)

    def get_persian_news_item(dir_addr, fname):
        f_addr = os.path.join(dir_addr, fname)
        f = open(f_addr, encoding="utf8")
        reader = csv.reader(f)
        try:
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                item = row[0].split('\t')
                try:
                    yield {"text": item[1], "label": item[2], "label_id": item[3]}
                except IndexError:
                    continue
        except GeneratorExit:
            f.close()

    if not is_exist_dataset(DATASET_INFO, dest_dir):
        downloaded_file = os.path.join(dest_dir, 'persian_news.zip')
        download_dataset(URL, dest_dir, filename=downloaded_file)
        dest_dir = unzip_dataset(downloaded_file, dest_dir, zip_format='zip')
    info = DatasetInfo(info_addr=info_addr)
    train_iterator = get_persian_news_item(dest_dir, 'persian_news/train.csv')
    test_iterator = get_persian_news_item(dest_dir, 'persian_news/test.csv')
    dev_iterator = get_persian_news_item(dest_dir, 'persian_news/dev.csv')
    sizes = DATASET_INFO['size']
    train_iterator = BaseIterator(train_iterator, num_lines=sizes['train'])
    test_iterator = BaseIterator(test_iterator, num_lines=sizes['test'])
    dev_iterator = BaseIterator(dev_iterator, num_lines=sizes['dev'])
    iterators = {'train': train_iterator, 'test': test_iterator, 'dev': dev_iterator}
    dataset = BaseDataset(info)
    dataset.set_iterators(iterators)
    return dataset

