# import os
# import json
# import pandas as pd
# from dadmatools.datasets.base import BaseDataset, DatasetInfo, BaseIterator
from dadmatools.datasets.dataset_utils import download_dataset, is_exist_dataset, DEFAULT_CACHE_DIR, unzip_dataset

# URL = 'https://github.com/Arman-Rayan-Sharif/arman-text-emotion.git'
DATASET_NAME = "armanRayanSentiment"

# def ArmanRayanSentiment(dest_dir=DEFAULT_CACHE_DIR):
#     base_addr = os.path.dirname(__file__)
#     info_addr = os.path.join(base_addr, 'info.py')
#     DATASET_INFO = json.load(open(info_addr))
#     dest_dir = os.path.join(dest_dir, DATASET_NAME)

#     def get_ate_item(dir_addr, fname):
#         f_addr = os.path.join(dir_addr, fname)
#         keys = ['text', 'label']
#         df = pd.read_table(f_addr, header=None)
#         for index, row in df.iterrows():
#             yield {k: row[i] for i, k in enumerate(keys)}

#     if not is_exist_dataset(DATASET_INFO, dest_dir):
#         downloaded_file = os.path.join(dest_dir, 'arman_text_emotion.zip')
#         download_dataset(URL, dest_dir, filename=downloaded_file)
#         dest_dir = unzip_dataset(downloaded_file, dest_dir, zip_format='zip')

#     info = DatasetInfo(info_addr=info_addr)
#     train_iterator = get_ate_item(dest_dir, 'arman-text-emotion/dataset/train.tsv')
#     test_iterator = get_ate_item(dest_dir, 'arman-text-emotion/dataset/test.tsv')
#     dev_iterator = get_ate_item(dest_dir, 'arman-text-emotion/dataset/dev.tsv')
#     sizes = DATASET_INFO['size']
#     train_iterator = BaseIterator(train_iterator, num_lines=sizes['train'])
#     test_iterator = BaseIterator(test_iterator, num_lines=sizes['test'])
#     dev_iterator = BaseIterator(dev_iterator, num_lines=sizes['dev'])
#     iterators = {'train': train_iterator, 'test': test_iterator, 'dev': dev_iterator}
#     dataset = BaseDataset(info)
#     dataset.set_iterators(iterators)
#     return dataset


# import os
# import pandas as pd
# import json
# from dadmatools.datasets.base import BaseDataset, DatasetInfo, BaseIterator

# # DATASET_NAME = "armanTextEmotion"
# DATA_PATH = './arman-text-emotion/dataset'

# LABEL_DICT = {
#     'OTHER': 0,
#     'HAPPY': 1,
#     'SURPRISE': 2,
#     'FEAR': 3,
#     'HATE': 4,
#     'ANGRY': 5,
#     'SAD': 6,
# }

# def ArmanRayanSentiment(dest_dir=DEFAULT_CACHE_DIR):
#     # Clone the dataset repository if not already cloned
#     if not os.path.exists(DATA_PATH):
#         os.system('git clone https://github.com/Arman-Rayan-Sharif/arman-text-emotion.git')

#     # Install required libraries
#     os.system('pip install transformers')

#     # Load dataset information
#     base_addr = os.path.dirname(__file__)
#     info_addr = os.path.join(base_addr, 'info.json')  # Assuming you have an info.json file for dataset info
#     if not os.path.exists(info_addr):
#         dataset_info = {
#             "name": DATASET_NAME,
#             "description": "Arman Text Emotion Dataset",
#             "size": {"train": 0, "test": 0, "dev": 0}  # Update these sizes accordingly
#         }
#         with open(info_addr, 'w') as f:
#             json.dump(dataset_info, f)

#     DATASET_INFO = json.load(open(info_addr))
#     dest_dir = os.path.join(dest_dir, DATASET_NAME)

#     def get_item(df):
#         for index, row in df.iterrows():
#             yield {'text': row[0], 'label': LABEL_DICT[row[1]]}

#     def load_and_split_data():
#         # Load the training and test datasets
#         train_df = pd.read_table(f'{DATA_PATH}/train.tsv', header=None)
#         test_df = pd.read_table(f'{DATA_PATH}/test.tsv', header=None)

#         # Create dev set from train set (10% of train set)
#         dev_fraction = 0.1
#         dev_df = train_df.sample(frac=dev_fraction, random_state=42)
#         train_df.drop(dev_df.index, inplace=True)

#         return train_df, test_df, dev_df

#     train_df, test_df, dev_df = load_and_split_data()

#     # Save sizes in the dataset info
#     DATASET_INFO['size']['train'] = len(train_df)
#     DATASET_INFO['size']['test'] = len(test_df)
#     DATASET_INFO['size']['dev'] = len(dev_df)

#     # Save the data to text files
#     os.makedirs(dest_dir, exist_ok=True)
#     train_file = os.path.join(dest_dir, 'train.txt')
#     test_file = os.path.join(dest_dir, 'test.txt')
#     dev_file = os.path.join(dest_dir, 'dev.txt')

#     def save_to_file(dataframe, filepath):
#         with open(filepath, 'w', encoding='utf8') as file:
#             for index, row in dataframe.iterrows():
#                 file.write(f"{row[0]}\t{LABEL_DICT[row[1]]}\n")

#     save_to_file(train_df, train_file)
#     save_to_file(test_df, test_file)
#     save_to_file(dev_df, dev_file)

#     info = DatasetInfo(info_addr=info_addr)
#     train_iterator = BaseIterator(get_item(train_df), num_lines=DATASET_INFO['size']['train'])
#     test_iterator = BaseIterator(get_item(test_df), num_lines=DATASET_INFO['size']['test'])
#     dev_iterator = BaseIterator(get_item(dev_df), num_lines=DATASET_INFO['size']['dev'])
#     iterators = {'train': train_iterator, 'test': test_iterator, 'dev': dev_iterator}
#     dataset = BaseDataset(info)
#     dataset.set_iterators(iterators)
#     return dataset

import os
import pandas as pd
import json
from dadmatools.datasets.base import BaseDataset, DatasetInfo, BaseIterator
from dadmatools.datasets.dataset_utils import download_dataset, is_exist_dataset, DEFAULT_CACHE_DIR, unzip_dataset

# URL = 'https://github.com/Arman-Rayan-Sharif/arman-text-emotion.git'
DATASET_NAME = "armanRayanSentiment"

DATA_PATH = './arman-text-emotion/dataset'

def ArmanRayanSentiment(dest_dir=DEFAULT_CACHE_DIR):
    # Clone the dataset repository if not already cloned
    if not os.path.exists(DATA_PATH):
        os.system('git clone https://github.com/Arman-Rayan-Sharif/arman-text-emotion.git')

    # Install required libraries
    os.system('pip install transformers')

    # Load dataset information
    base_addr = os.path.dirname(__file__)
    info_addr = os.path.join(base_addr, 'info.json')  # Assuming you have an info.json file for dataset info
    if not os.path.exists(info_addr):
        dataset_info = {
            "name": DATASET_NAME,
            "description": "Arman Text Emotion Dataset",
            "size": {"train": 0, "test": 0, "dev": 0}  # Update these sizes accordingly
        }
        with open(info_addr, 'w') as f:
            json.dump(dataset_info, f)

    DATASET_INFO = json.load(open(info_addr))
    dest_dir = os.path.join(dest_dir, DATASET_NAME)

    def get_item(df):
        for index, row in df.iterrows():
            yield {'text': row[0], 'label': row[1]}

    def load_and_split_data():
        # Load the training and test datasets
        train_df = pd.read_table(f'{DATA_PATH}/train.tsv', header=None)
        test_df = pd.read_table(f'{DATA_PATH}/test.tsv', header=None)

        # Create dev set from train set (10% of train set)
        dev_fraction = 0.1
        dev_df = train_df.sample(frac=dev_fraction, random_state=42)
        train_df.drop(dev_df.index, inplace=True)

        return train_df, test_df, dev_df

    train_df, test_df, dev_df = load_and_split_data()

    # Save sizes in the dataset info
    DATASET_INFO['size']['train'] = len(train_df)
    DATASET_INFO['size']['test'] = len(test_df)
    DATASET_INFO['size']['dev'] = len(dev_df)

    # Save the data to text files
    os.makedirs(dest_dir, exist_ok=True)
    train_file = os.path.join(dest_dir, 'train.txt')
    test_file = os.path.join(dest_dir, 'test.txt')
    dev_file = os.path.join(dest_dir, 'dev.txt')

    def save_to_file(dataframe, filepath):
        with open(filepath, 'w', encoding='utf8') as file:
            for index, row in dataframe.iterrows():
                file.write(f"{row[0]}\t{row[1]}\n")

    save_to_file(train_df, train_file)
    save_to_file(test_df, test_file)
    save_to_file(dev_df, dev_file)

    info = DatasetInfo(info_addr=info_addr)
    train_iterator = BaseIterator(get_item(train_df), num_lines=DATASET_INFO['size']['train'])
    test_iterator = BaseIterator(get_item(test_df), num_lines=DATASET_INFO['size']['test'])
    dev_iterator = BaseIterator(get_item(dev_df), num_lines=DATASET_INFO['size']['dev'])
    iterators = {'train': train_iterator, 'test': test_iterator, 'dev': dev_iterator}
    dataset = BaseDataset(info)
    dataset.set_iterators(iterators)
    return dataset

