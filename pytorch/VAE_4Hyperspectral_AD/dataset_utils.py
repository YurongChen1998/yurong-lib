from random import sample
import bisect
import warnings
from torch.utils.data import Dataset

class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

def get_random_subset(names, labels, percent):
    """

    :param names: list of names
    :param labels:  list of labels
    :param percent: 0 < float < 1
    :return:
    """
    samples = len(names)
    amount = int(samples * percent)
    random_index = sample(range(samples), amount)
    name_val = [names[k] for k in random_index]
    name_train = [v for k, v in enumerate(names) if k not in random_index]
    labels_val = [labels[k] for k in random_index]
    labels_train = [v for k, v in enumerate(labels) if k not in random_index]
    return name_train, name_val, labels_train, labels_val

def get_dataset_info(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels

def get_split_dataset_info(txt_list, val_percentage):
    names, labels = get_dataset_info(txt_list)
    return get_random_subset(names, labels, val_percentage)
