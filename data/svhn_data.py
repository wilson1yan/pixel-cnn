"""
Utilities for downloading and unpacking the SVHN dataset
"""

import os
import sys
import tarfile
from six.moves import urllib
import numpy as np
import scipy.io

URL = 'http://ufldl.stanford.edu/housenumbers/{}_32x32.mat'

def maybe_download_and_extract(data_dir, subset):
    url = URL.format(subset)
    filename = url.split('/')[-1]
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

def unpickle(file):
    fo = open(file, 'rb')
    if (sys.version_info >= (3, 0)):
        import pickle
        d = pickle.load(fo, encoding='latin1')
    else:
        import cPickle
        d = cPickle.load(fo)
    fo.close()
    return {'x': d['data'].reshape((10000,3,32,32)), 'y': np.array(d['labels']).astype(np.uint8)}

def load(data_dir, subset='train'):
    maybe_download_and_extract(data_dir, subset)
    if subset=='train':
        train_data = scipy.io.loadmat(os.path.join(data_dir, 'train_32x32.mat'))
        trainx = train_data['X'].transpose((3, 2, 0, 1))
        trainy = train_data['y'].squeeze(1)
        return trainx, trainy
    elif subset=='test':
        test_data = scipy.io.loadmat(os.path.join(data_dir, 'test_32x32.mat'))
        testx = test_data['X'].transpose((3, 2, 0, 1))
        testy = test_data['y'].squeeze(1)
        return testx, testy
    else:
        raise NotImplementedError('subset should be either train or test')

class DataLoader(object):
    """ an object that generates batches of SVHN data for training """

    def __init__(self, data_dir, subset, batch_size, rng=None, shuffle=False, return_labels=False):
        """ 
        - data_dir is location where to store files
        - subset is train|test 
        - batch_size is int, of #examples to load at once
        - rng is np.random.RandomState object for reproducibility
        """

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_labels = return_labels

        # create temporary storage for the data, if not yet created
        if not os.path.exists(data_dir):
            print('creating folder', data_dir)
            os.makedirs(data_dir)

        # load SVHN training data to RAM
        self.data, self.labels = load(data_dir, subset=subset)
        self.data = np.transpose(self.data, (0,2,3,1)) # (N,3,32,32) -> (N,32,32,3)
        
        self.p = 0 # pointer to where we are in iteration
        self.rng = np.random.RandomState(1) if rng is None else rng

    def get_observation_size(self):
        return self.data.shape[1:]

    def get_num_labels(self):
        return np.amax(self.labels) + 1

    def reset(self):
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        if n is None: n = self.batch_size

        # on first iteration lazily permute all data
        if self.p == 0 and self.shuffle:
            inds = self.rng.permutation(self.data.shape[0])
            self.data = self.data[inds]
            self.labels = self.labels[inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.data.shape[0]:
            self.reset() # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.data[self.p : self.p + n]
        y = self.labels[self.p : self.p + n]
        self.p += self.batch_size

        if self.return_labels:
            return x,y
        else:
            return x

    next = __next__  # Python 2 compatibility (https://stackoverflow.com/questions/29578469/how-to-make-an-object-both-a-python2-and-python3-iterator)


