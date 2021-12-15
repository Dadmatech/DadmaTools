import json
from collections import Iterator


class DatasetInfo:
    def __init__(self, info_addr):
        with open(info_addr) as f:
            info = json.load(f)
            self.version = info['version']
            self.task = info['task']
            self.size = info['size']
            self.splits = info['splits']

class BaseDataset(Iterator):

    def __init__(self, iterator, info, num_lines=1000000000):
        self.iterator = iterator
        self.info = info
        self.num_lines = num_lines
        self.current_pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_pos == self.num_lines - 1:
            raise StopIteration
        item = next(self.iterator)
        if self.current_pos is None:
            self.current_pos = 0
        else:
            self.current_pos += 1
        return item

    def __len__(self):
        return self.num_lines

    def pos(self):
        """
        Returns current position of the iterator. This returns None
        if the iterator hasn't been used yet.
        """
        return self.current_pos


class SplittedDataset:
    """Abstract dataset class for dataset-like object, like list and array.
    All datasets(sub-classes) should inherit.
    Args:
        data (list, array, tuple): dataset-like object
    """

    def __init__(self, train=None, test=None, dev=None):
        self.train = train
        self.test = test
        self.dev = dev
