import os
import pandas as pd
import json
import gdown
import numpy as np
from dadmatools.datasets.base import BaseDataset, DatasetInfo, BaseIterator
from dadmatools.datasets.dataset_utils import download_dataset, is_exist_dataset, DEFAULT_CACHE_DIR, unzip_dataset


# Define dataset parameters
DATASET_NAME = "persian3labels"
URL = 'https://drive.google.com/uc?id=1l1s6mmGMZUZBfW7tdbbnDzQw33eOrQIj'
DATA_PATH = './persian3labels/dataset'
def Persian3labels(dest_dir=DEFAULT_CACHE_DIR):
    dest_dir = os.path.join(dest_dir, DATASET_NAME)

    # Download the dataset from Google Drive
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        output = os.path.join(dest_dir, 'persian_sentiment.csv')
        gdown.download(URL, output, quiet=False)

    # Load the dataset into a DataFrame
    df = pd.read_csv(os.path.join(dest_dir, 'persian_sentiment.csv'))

    def get_item(df):
        for index, row in df.iterrows():
            yield {'text': row['comment'], 'label': row['sentiment']}

    def save_to_file(dataframe, filepath):
        with open(filepath, 'w', encoding='utf8') as file:
            for index, row in dataframe.iterrows():
                file.write(f"{row['comment']}\t{row['sentiment']}\n")

    # Split the dataset into train, test, and dev sets
    train_df, dev_df, test_df = np.split(df.sample(frac=1, random_state=42), [int(.8*len(df)), int(.9*len(df))])

    # Save the data to text files
    train_file = os.path.join(dest_dir, 'train.txt')
    test_file = os.path.join(dest_dir, 'test.txt')
    dev_file = os.path.join(dest_dir, 'dev.txt')

    save_to_file(train_df, train_file)
    save_to_file(test_df, test_file)
    save_to_file(dev_df, dev_file)

    # Create dataset info
    dataset_info = {
        "name": DATASET_NAME,
        "description": "Persian Sentiment Dataset",
        "size": {
            "train": len(train_df),
            "test": len(test_df),
            "dev": len(dev_df)
        }
    }

    info_addr = os.path.join(dest_dir, 'info.json')
    with open(info_addr, 'w') as f:
        json.dump(dataset_info, f)

    DATASET_INFO = dataset_info

    info = DatasetInfo(info_addr=info_addr)
    train_iterator = BaseIterator(get_item(train_df), num_lines=DATASET_INFO['size']['train'])
    test_iterator = BaseIterator(get_item(test_df), num_lines=DATASET_INFO['size']['test'])
    dev_iterator = BaseIterator(get_item(dev_df), num_lines=DATASET_INFO['size']['dev'])
    iterators = {'train': train_iterator, 'test': test_iterator, 'dev': dev_iterator}
    dataset = BaseDataset(info)
    dataset.set_iterators(iterators)
    return dataset


