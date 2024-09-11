import csv
import os
import json
from dadmatools.datasets.base import BaseDataset, DatasetInfo, BaseIterator
from dadmatools.datasets.dataset_utils import download_dataset, is_exist_dataset, DEFAULT_CACHE_DIR, unzip_dataset

# Google Drive URL for your dataset
URL = 'https://drive.google.com/uc?id=1PRaDr9XkpGIN20WqCnOH2pl4AU8aCEVi'
DATASET_NAME = "persianDataset"

def PersianSentiment(dest_dir=DEFAULT_CACHE_DIR):
    base_addr = os.path.dirname(__file__)
    info_addr = os.path.join(base_addr, 'info.json')  # Changed to JSON

    # Load the info.json file
    with open(info_addr, 'r', encoding='utf-8') as f:
        DATASET_INFO = json.load(f)

    dest_dir = os.path.join(dest_dir, DATASET_NAME)

    def get_persian_sa_item(dir_addr, fname):
        """Iterates through the dataset files."""
        f_addr = os.path.join(dir_addr, fname)
        keys = ['index', 'text', 'label', 'label_id']
        with open(f_addr, encoding="utf8") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    continue  # Skip header
                item = row[0].split('\t')
                if len(item) < 2 or not item[1].strip():  # Skip if text is empty
                    continue
                try:
                    yield {k: item[i] for i, k in enumerate(keys)}
                except IndexError:
                    continue

    if not is_exist_dataset(DATASET_INFO, dest_dir):
        downloaded_file = os.path.join(dest_dir, 'persian_dataset.zip')
        download_dataset(URL, dest_dir, filename=downloaded_file)
        dest_dir = unzip_dataset(downloaded_file, dest_dir, zip_format='zip')

    info = DatasetInfo(info_addr=info_addr)

    # Creating iterators for train, test, and dev datasets
    train_iterator = get_persian_sa_item(dest_dir, 'tenLabeled/train.csv')
    test_iterator = get_persian_sa_item(dest_dir, 'tenLabeled/test.csv')
    dev_iterator = get_persian_sa_item(dest_dir, 'tenLabeled/dev.csv')

    # Fetch the dataset sizes from info
    sizes = DATASET_INFO['size']

    # Creating BaseIterators
    train_iterator = BaseIterator(train_iterator, num_lines=sizes['train'])
    test_iterator = BaseIterator(test_iterator, num_lines=sizes['test'])
    dev_iterator = BaseIterator(dev_iterator, num_lines=sizes['dev'])

    # Packaging iterators into a dictionary
    iterators = {'train': train_iterator, 'test': test_iterator, 'dev': dev_iterator}

    # Defining tagset (labels)
    tagset = list(DATASET_INFO['splits'])  # Make sure this represents labels if required

    # Creating the BaseDataset object and returning it
    dataset = BaseDataset(info, tagset)
    dataset.set_iterators(iterators)

    return dataset
