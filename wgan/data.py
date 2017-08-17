'''
Dataset loaders.
'''

from os import path
import torch
from torch.utils.data import Dataset
import h5py #pylint: disable=unused-import
import h5py_cache

class Cifar10Dataset(Dataset):
    def __init__(self, data_dir, subset='train'):
        '''Creates a Dataset object for loading CIFAR-10 data.

        It is expected that the data has been downloaded and preprocessed using
        [DLDS](https://github.com/anibali/dlds).

        Args:
            data_dir: path to the directory containing `cifar-10.h5`
            subset: subset of the data to load ("train" or "test")
        '''

        super().__init__()

        h5_file = path.join(data_dir, 'cifar-10.h5')
        self.h5_file = h5_file
        self.subset = subset
        with h5py_cache.File(h5_file, 'r', chunk_cache_mem_size=1024**3) as f:
            self.length = f[subset]['images'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        subset = self.subset

        with h5py_cache.File(self.h5_file, 'r', chunk_cache_mem_size=1024**3) as f:
            image_bytes = torch.from_numpy(f[subset]['images'][index])
            label = torch.from_numpy(f[subset]['labels'][index])

        image = (image_bytes.float() * 2) / 255 - 1

        sample = {
            'input': image,
            'label': label,
        }

        return sample
