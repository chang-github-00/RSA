import pickle as pkl
import lmdb

from torch.utils.data import Dataset


class LMDBDataset(Dataset):
    def __init__(self, data_file, in_memory):
        env = lmdb.open(data_file, max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))

        if in_memory:
            cache = [None] * num_examples
            self._cache = cache

        self._env = env
        self._in_memory = in_memory
        self._num_examples = num_examples

    def __len__(self):
        return self._num_examples

    def __getitem__(self, index):
        if self._in_memory and self._cache[index] is not None:
            item = self._cache[index]
        else:
            with self._env.begin(write=False) as txn:
                item = pkl.loads(txn.get(str(index).encode()))
                if 'id' not in item:
                    item['id'] = str(index)
                if self._in_memory:
                    self._cache[index] = item
        return item


class DataProcessor:
    """Base class for data converters for biological tasks data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

