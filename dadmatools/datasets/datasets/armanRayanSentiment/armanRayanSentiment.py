import os
import json
import csv
from dadmatools.datasets.base import BaseDataset, DatasetInfo, BaseIterator
from dadmatools.datasets.dataset_utils import download_dataset, is_exist_dataset, DEFAULT_CACHE_DIR, unzip_dataset

# URL to download the Arman Emo dataset (Google Drive link you provided)
URL = 'https://drive.google.com/uc?id=1bkjtXrHbiW0Z9-1ADs9Qh-dPYHwX1VRa'
DATASET_NAME = "armanRayanSentiment"

def ArmanRayanSentiment(dest_dir=DEFAULT_CACHE_DIR):
    base_addr = os.path.dirname(__file__)
    info_addr = os.path.join(base_addr, 'info.py')  # Ensure 'info.py' exists
    DATASET_INFO = json.load(open(info_addr))  # Load dataset info
    dest_dir = os.path.join(dest_dir, DATASET_NAME)

    # Function to read and yield items from CSV files
    def get_arman_item(dir_addr, fname):
        f_addr = os.path.join(dir_addr, fname)
        keys = ['index', 'text', 'label', 'label_id']  # The columns in the CSV files
        with open(f_addr, encoding="utf8") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:  # Skip header
                    continue
                item = row[0].split('\t')  # Assuming tab-separated format
                try:
                    yield {k: item[i] for i, k in enumerate(keys)}  # Yield item as dictionary
                except IndexError:
                    continue  # Skip rows with missing values

    # Check if dataset is already downloaded and extracted
    if not is_exist_dataset(DATASET_INFO, dest_dir):
        downloaded_file = os.path.join(dest_dir, 'arman_sentiment.zip')
        download_dataset(URL, dest_dir, filename=downloaded_file)
        dest_dir = unzip_dataset(downloaded_file, dest_dir, zip_format='zip')

    info = DatasetInfo(info_addr=info_addr)

    # Set iterators for train, test, and dev
    train_iterator = get_arman_item(dest_dir, 'train.csv')
    test_iterator = get_arman_item(dest_dir, 'test.csv')
    dev_iterator = get_arman_item(dest_dir, 'dev.csv')

    # Get dataset sizes from info.py
    sizes = DATASET_INFO['size']
    train_iterator = BaseIterator(train_iterator, num_lines=sizes['train'])
    test_iterator = BaseIterator(test_iterator, num_lines=sizes['test'])
    dev_iterator = BaseIterator(dev_iterator, num_lines=sizes['dev'])

    # Define iterators and tagset (label dictionary)
    tagset = ['SAD', 'HAPPY', 'SURPRISE', 'HATE', 'FEAR', 'ANGRY', 'OTHER']
    iterators = {'train': train_iterator, 'test': test_iterator, 'dev': dev_iterator}

    # Create and return dataset object
    dataset = BaseDataset(info, tagset)
    dataset.set_iterators(iterators)
    return dataset
