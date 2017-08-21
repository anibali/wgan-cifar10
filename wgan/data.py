'''
Dataset loaders.
'''

import os
import urllib.request
import hashlib
import torch
from torch.utils.data import Dataset
import h5py #pylint: disable=unused-import
import h5py_cache

from wgan import progress

class Cifar10Dataset(Dataset):
    '''Creates a Dataset object for loading CIFAR-10 data from a HDF5 file.

    Args:
        data_dir (str): path to the directory containing `cifar-10.h5`
        subset (str): subset of the data to load ("train" or "test")
        download (bool): downloads required data when True
    '''

    def __init__(self, data_dir, subset='train', download=True):
        super().__init__()

        if download:
            Cifar10Dataset.download(data_dir)

        h5_file = os.path.join(data_dir, 'cifar-10.h5')
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

    @staticmethod
    def check(data_dir):
        '''Verify the presence and integrity of data.'''

        h5_file = os.path.join(data_dir, 'cifar-10.h5')

        if not os.path.isfile(h5_file):
            return False

        expected_md5sum = '171c56987005742c40207c662f4bddc7'
        md5 = hashlib.md5()
        with open(h5_file, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                md5.update(chunk)
        if md5.hexdigest() != expected_md5sum:
            return False

        return True

    @staticmethod
    def download(data_dir):
        '''Download a HDF5 file containing the CIFAR-10 dataset.'''

        # Do nothing if we already have the data
        if Cifar10Dataset.check(data_dir):
            return

        print('Downloading CIFAR-10 dataset...')
        os.makedirs(data_dir, exist_ok=True)

        src_url = 'https://s3-us-west-2.amazonaws.com/anibali-dl/cifar-10.h5'
        dest_file = os.path.join(data_dir, 'cifar-10.h5')

        with urllib.request.urlopen(src_url) as src, open(dest_file, 'wb') as dest:
            total_bytes = int(src.info()['Content-Length'])
            downloaded_bytes = 0
            while True:
                buf = src.read(16*1024)
                if not buf:
                    break
                dest.write(buf)
                downloaded_bytes += len(buf)
                progress.bar(downloaded_bytes, total_bytes,
                    prefix='cifar-10.h5',
                    suffix='of {:0.1f} MiB'.format(total_bytes / 1024**2),
                    length=30)
