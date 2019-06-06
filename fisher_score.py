import os
import json
import argparse

import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--data_dir', type=str, default='/local_home/tim/pxpp/data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='/local_home/tim/pxpp/save', help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--data_set', type=str, default='cifar', help='Can be either cifar|imagenet|svhn|mnist')
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':')))

rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)

# initialize data loaders for train/test splits
if args.data_set == 'imagenet' and args.class_conditional:
    raise("We currently don't have labels for the small imagenet data set")
if args.data_set == 'cifar':
    import data.cifar10_data as cifar10_data
    DataLoader = cifar10_data.DataLoader
elif args.data_set == 'imagenet':
    import data.imagenet_data as imagenet_data
    DataLoader = imagenet_data.DataLoader
elif args.data_set == 'svhn':
    import data.svhn_data as svhn_data
    DataLoader = svhn_data.DataLoader
elif args.data_set == 'mnist':
    import data.mnist_data as mnist_data
    DataLoader = mnist_data.DataLoader
else:
    raise("unsupported dataset")
train_data = DataLoader(args.data_dir, 'train', args.batch_size * args.nr_gpu, rng=rng, shuffle=True, return_labels=args.class_conditional)
test_data = DataLoader(args.data_dir, 'test', args.batch_size * args.nr_gpu, shuffle=False, return_labels=args.class_conditional)
obs_shape = train_data.get_observation_size() # e.g. a tuple (32,32,3)
assert len(obs_shape) == 3, 'assumed right now'

sess = tf.Session()
ckpt_file = args.save_dir + '/params_' + args.data_set + '.ckpt'
saver = tf.train.import_meta_graph(ckpt_file + '.meta')
saver.restore(sess, ckpt_file)

