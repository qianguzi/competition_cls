from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os, cv2
import math, h5py
import numpy as np
import tensorflow as tf

import preprocess.first as first
import preprocess.default as default

flags = tf.app.flags
#/media/deeplearning/f3cff4c9-1ab9-47f0-8b82-231dedcbd61b/lcz
flags.DEFINE_string('dataset_folder', '/home/data/lcz', 'Folder containing dataset.')
flags.DEFINE_string('output_dir', '/media/jun/data/lcz/tfrecord', 'Output location of dataset.')
flags.DEFINE_string('preprocess_method', 'default', 'The image data preprocess term.')
FLAGS = flags.FLAGS

_NUM_CLASSES = 17
_NUM_SHARDS = 4
_PREPROCESS_METHOD = {
    'default': default.default_preprocess,
    'first': first.first_preprocess,
}

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def convert_dataset(dataset, s1, s2, labels, preprocess_fn):
  sys.stdout.write('Processing ' + dataset + '\n')
  sys.stdout.flush()

  num_samples = int(labels.shape[0])
  sys.stdout.write('Number of samples: %d\n' % (num_samples))
  sys.stdout.flush()

  num_per_shard = int(math.ceil(num_samples / float(_NUM_SHARDS)))
  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        FLAGS.output_dir,
        FLAGS.preprocess_method,
        '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
    writer = tf.python_io.TFRecordWriter(output_filename)
    start_idx = shard_id * num_per_shard
    end_idx = min((shard_id + 1) * num_per_shard, num_samples)
    for idx in range(start_idx, end_idx):
      sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
          idx + 1, num_samples, shard_id))
      sys.stdout.flush()
      s1_data = s1[idx]
      s2_data = s2[idx]

      label = int(np.argmax(labels[idx])+1)
      img_data = preprocess_fn(s1_data, s2_data)
      img_data = np.reshape(img_data,[-1]).astype(np.float32)

      example = tf.train.Example(features=tf.train.Features(feature={
          'data': _float_feature(img_data),
          'label': _int64_feature(label),
          'idx': _int64_feature(int(idx))
      }))
      writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()

def convert_dataset_balance(dataset, s1, s2, labels, preprocess_fn):
  sys.stdout.write('Processing ' + dataset + '\n')
  sys.stdout.flush()
  class_num = np.sum(labels, axis=0, dtype=np.uint16)

  dataset_idx = []
  per_class_num = np.max(class_num)
  for i in range(_NUM_CLASSES):
    dataset_idx.append(np.where(labels[:,i])[0])

  sys.stdout.write('Number of samples: %d\n' % (_NUM_CLASSES*per_class_num))
  sys.stdout.flush()

  num_per_shard = int(math.ceil(per_class_num / float(_NUM_SHARDS)))
  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        FLAGS.output_dir,
        FLAGS.preprocess_method,
        '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
    writer = tf.python_io.TFRecordWriter(output_filename)
    start_idx = shard_id * num_per_shard
    end_idx = min((shard_id + 1) * num_per_shard, per_class_num)
    for i in range(start_idx, end_idx):
      sys.stdout.write('\r>> Converting batch of images %d/%d shard %d' % (
          i + 1, per_class_num, shard_id))
      sys.stdout.flush()
      for j in range(_NUM_CLASSES):
        try:
          idx = dataset_idx[j][i]
        except:
          idx = dataset_idx[j][np.random.randint(class_num[j])]

        label = int(np.argmax(labels[idx])+1)
        if label != (j+1):
          raise RuntimeError('Label is wrong.')
        s1_data = s1[idx]
        s2_data = s2[idx]
        img_data = preprocess_fn(s1_data, s2_data)
        img_data = np.reshape(img_data, [-1]).astype(np.float32)

        example = tf.train.Example(features=tf.train.Features(feature={
            'data': _float_feature(img_data),
            'label': _int64_feature(label),
            'idx': _int64_feature(int(idx))
        }))
        writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def main():
  path_training = os.path.join(FLAGS.dataset_folder, 'training.h5')
  path_validation = os.path.join(FLAGS.dataset_folder, 'validation.h5')
  fid_training = h5py.File(path_training,'r')
  s1_training = fid_training['sen1']
  s2_training = fid_training['sen2']
  label_training = fid_training['label']
  fid_validation = h5py.File(path_validation,'r')
  s1_validation = fid_validation['sen1']
  s2_validation = fid_validation['sen2']
  label_validation = fid_validation['label']

  if FLAGS.preprocess_method not in _PREPROCESS_METHOD:
    raise ValueError('The specified preprocess method is not supported yet.')
  preprocess_fn = _PREPROCESS_METHOD[FLAGS.preprocess_method]
  tf.gfile.MakeDirs(FLAGS.output_dir)
  tf.gfile.MakeDirs(os.path.join(FLAGS.output_dir, FLAGS.preprocess_method))

  convert_dataset_balance('train', s1_training, s2_training, label_training, preprocess_fn)

  convert_dataset('val', s1_validation, s2_validation, label_validation, preprocess_fn)

if __name__ == '__main__':
  main()