import os
import pandas as pd
import json
import gdown
import numpy as np
from dadmatools.datasets.base import BaseDataset, DatasetInfo, BaseIterator
from dadmatools.datasets.dataset_utils import download_dataset, is_exist_dataset, DEFAULT_CACHE_DIR, unzip_dataset

# Define dataset parameters
DATASET_NAME = "persianDataset"
URL = 'https://drive.google.com/uc?id=1p94KROlL5WqirQ9M84em8C4JlXGcAuu0'

def PersianDataset(dest_dir=DEFAULT_CACHE_DIR):
    dest_dir = os.path.join(dest_dir, DATASET_NAME)

    # Download the dataset from Google Drive
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        output = os.path.join(dest_dir, 'persian_dataset.csv')
        gdown.download(URL, output, quiet=False)

    # Load the dataset into a DataFrame
    df = pd.read_csv(os.path.join(dest_dir, 'persian_dataset.csv'))
    
    # Validate expected columns
    expected_columns = ['text', 'sentiment', 'emotion', 'confidence']
    for col in expected_columns:
        if col not in df.columns:
            raise ValueError(f"Expected column {col} not found in the dataset")

    def save_to_file(dataframe, filepath, text_column, label_column):
        with open(filepath, 'w', encoding='utf8') as file:
            for index, row in dataframe.iterrows():
                file.write(f"{row[text_column]}\t{row[label_column]}\n")

    # Split the dataset into train, test, and dev sets (80%, 10%, 10%)
    train_df, dev_df, test_df = np.split(df.sample(frac=1, random_state=42), 
                                         [int(.8*len(df)), int(.9*len(df))])

    # Define folder structure
    label_types = ['threeLabeled', 'tenLabeled']

    # Creating the necessary directories and saving the files
    for label_type in label_types:
        sub_dir = os.path.join(dest_dir, label_type)
        os.makedirs(sub_dir, exist_ok=True)

        if label_type == 'threeLabeled':
            train_df['label'] = train_df['sentiment']
            dev_df['label'] = dev_df['sentiment']
            test_df['label'] = test_df['sentiment']
        else:  # 'tenLabeled'
            train_df['label'] = train_df['emotion']
            dev_df['label'] = dev_df['emotion']
            test_df['label'] = test_df['emotion']

        # File paths
        train_file = os.path.join(sub_dir, 'train.txt')
        test_file = os.path.join(sub_dir, 'test.txt')
        dev_file = os.path.join(sub_dir, 'dev.txt')

        # Save the data to text files
        save_to_file(train_df, train_file, 'text', 'label')
        save_to_file(test_df, test_file, 'text', 'label')
        save_to_file(dev_df, dev_file, 'text', 'label')

    # Create dataset info
    dataset_info = {
        "name": DATASET_NAME,
        "description": "Persian Sentiment and Emotion Dataset",
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

    # Create iterators for one of the configurations (e.g., three-labeled)
    info = DatasetInfo(info_addr=info_addr)
    train_iterator = BaseIterator(lambda: iter(train_df[['text', 'label']].itertuples(index=False, name=None)), 
                                  num_lines=DATASET_INFO['size']['train'])
    test_iterator = BaseIterator(lambda: iter(test_df[['text', 'label']].itertuples(index=False, name=None)), 
                                 num_lines=DATASET_INFO['size']['test'])
    dev_iterator = BaseIterator(lambda: iter(dev_df[['text', 'label']].itertuples(index=False, name=None)), 
                                num_lines=DATASET_INFO['size']['dev'])

    iterators = {'train': train_iterator, 'test': test_iterator, 'dev': dev_iterator}
    dataset = BaseDataset(info)
    dataset.set_iterators(iterators)
    return dataset


