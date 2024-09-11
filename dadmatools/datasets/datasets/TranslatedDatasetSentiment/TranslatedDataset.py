import os
import pandas as pd
import json
import gdown
import numpy as np
from dadmatools.datasets.base import BaseDataset, DatasetInfo, BaseIterator
from dadmatools.datasets.dataset_utils import download_dataset, is_exist_dataset, DEFAULT_CACHE_DIR, unzip_dataset

# Define dataset parameters
DATASET_NAME = "translatedDataset"
URL = 'https://drive.google.com/uc?id=1NHO0lqIZWypN6GVB6wZOWNcV7sGa3Ia3'

def TranslatedDataset(dest_dir=DEFAULT_CACHE_DIR):
    dest_dir = os.path.join(dest_dir, DATASET_NAME)

    # Download the dataset from Google Drive
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        output = os.path.join(dest_dir, 'translated_dataset.csv')
        gdown.download(URL, output, quiet=False)

    # Load the dataset into a DataFrame
    df = pd.read_csv(os.path.join(dest_dir, 'translated_dataset.csv'))

    def save_to_file(dataframe, filepath, text_column, label_column):
        with open(filepath, 'w', encoding='utf8') as file:
            for index, row in dataframe.iterrows():
                file.write(f"{row[text_column]}\t{row[label_column]}\n")

    # Split the dataset into train, test, and dev sets (80%, 10%, 10%)
    train_df, dev_df, test_df = np.split(df.sample(frac=1, random_state=42), 
                                         [int(.8*len(df)), int(.9*len(df))])

    # Define folder structure
    languages = ['persian', 'english']
    label_types = ['threeLabeled', 'fifthLabeled']
    text_columns = {'persian': 'Text_Farsi', 'english': 'Text_English'}
    label_columns = {'threeLabeled': 'Sentiment', 'fifthLabeled': 'Emotion'}

    # Create directories and save files
    for lang in languages:
        for label_type in label_types:
            sub_dir = os.path.join(dest_dir, lang, label_type)
            os.makedirs(sub_dir, exist_ok=True)

            text_column = text_columns[lang]
            label_column = label_columns[label_type]

            # File paths
            train_file = os.path.join(sub_dir, 'train.txt')
            test_file = os.path.join(sub_dir, 'test.txt')
            dev_file = os.path.join(sub_dir, 'dev.txt')

            # Save the data to text files
            save_to_file(train_df, train_file, text_column, label_column)
            save_to_file(test_df, test_file, text_column, label_column)
            save_to_file(dev_df, dev_file, text_column, label_column)

    # Create dataset info
    dataset_info = {
        "name": DATASET_NAME,
        "description": "Translated Sentiment and Emotion Dataset",
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

    # Create iterators for one of the configurations (e.g., Persian, three-labeled)
    info = DatasetInfo(info_addr=info_addr)
    train_iterator = BaseIterator(lambda: save_to_file(train_df, train_file, 'Text_Farsi', 'Sentiment'), 
                                  num_lines=DATASET_INFO['size']['train'])
    test_iterator = BaseIterator(lambda: save_to_file(test_df, test_file, 'Text_Farsi', 'Sentiment'), 
                                 num_lines=DATASET_INFO['size']['test'])
    dev_iterator = BaseIterator(lambda: save_to_file(dev_df, dev_file, 'Text_Farsi', 'Sentiment'), 
                                num_lines=DATASET_INFO['size']['dev'])
    iterators = {'train': train_iterator, 'test': test_iterator, 'dev': dev_iterator}
    dataset = BaseDataset(info)
    dataset.set_iterators(iterators)
    return dataset

# Use the function to process and save the dataset
dataset = TranslatedDataset()
