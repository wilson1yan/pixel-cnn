"""
Utilities for downloading and unpacking the MNIST dataset
"""

import os
import sys
from six.moves import urllib
import numpy as np
from mnist import MNIST

URL = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

def download(filename, url, filepath):
    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                         float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')


def maybe_download_and_extract(data_dir, subset):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    img_fname = TRAIN_IMAGES if subset == 'train' else TEST_IMAGES
    label_fname = TRAIN_LABELS if subset == 'train' else TEST_LABELS

    image_path = os.path.join(data_dir, img_fname)
    label_path = os.path.join(data_dir, label_fname)

    if not os.path.exists(image_path) or not os.path.exists(label_path):
        image_url = URL + img_fname
        label_url = URL + label_fname

        download(img_fname, image_url, image_path)
        download(label_fname, label_url, label_path)


def load(data_dir, subset='train'):
    maybe_download_and_extract(data_dir, subset)
    mndata = MNIST(data_dir)
    mndata.gz = True
    if subset=='train':
        trainx, trainy = mndata.load_training()
        trainx, trainy = np.array(trainx), np.array(trainy)
        trainx = trainx.reshape((trainx.shape[0], 1, 28, 28))
        return trainx, trainy
    elif subset=='test':
        testx, testy = mndata.load_testing()
        testx, testy = np.array(testx), np.array(testy)
        testx = testx.reshape((testx.shape[0], 1, 28, 28))
        return testx, testy
    else:
        raise NotImplementedError('subset should be either train or test')

class DataLoader(object):
    """ an object that generates batches of MNIST data for training """

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

        # load MNIST training data to RAM
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


