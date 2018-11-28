import sys
import os, cv2
import math, h5py
import numpy as np
import tensorflow as tf

import preprocess

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset_folder',
                           '/home/data/lcz',
                           'Folder containing dataset.')

tf.app.flags.DEFINE_string(
    'output_dir',
    '/media/jun/data/lcz/tfrecord',
    'Path to save converted SSTable of TensorFlow examples.')

_DATASET_MEAN = [6.79034058e-05, 3.13958198e-05, 8.54824102e-05, -7.53851136e-05,
                -1.61852048e+00, -1.02397158e+00, -3.99997315e-04, 1.44433013e-03,
                 1.29802515e-01, 1.17745393e-01, 1.14519432e-01, 1.27774508e-01,
                 1.70839563e-01, 1.92380020e-01, 1.85248741e-01, 2.06286210e-01,
                 1.74647946e-01, 1.28752703e-01]
_DATASET_STD = [0.20595731, 0.20523204, 0.48476867, 0.48547576, 0.49312326, 0.54128573,
                0.25078281, 0.18858326, 0.0413942, 0.05197171, 0.07318039, 0.06928833,
                0.07388366, 0.08290952, 0.08446056, 0.09017361, 0.09451134, 0.09073909]

_NUM_CLASSES = 17
_NUM_SHARDS = 4

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def convert_dataset(dataset, s1, s2, labels, num):
  sys.stdout.write('Processing ' + dataset + '\n')
  sys.stdout.flush()
  sys.stdout.write('Number of samples: %d\n' % (num))
  sys.stdout.flush()

  num_per_shard = int(math.ceil(num / float(_NUM_SHARDS)))
  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        FLAGS.output_dir,
        '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
    writer = tf.python_io.TFRecordWriter(output_filename)
    start_idx = shard_id * num_per_shard
    end_idx = min((shard_id + 1) * num_per_shard, num)
    for idx in range(start_idx, end_idx):
      sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
          idx + 1, num, shard_id))
      sys.stdout.flush()
      s1_data = s1[idx]
      s2_data = s2[idx]
      label = int(np.argmax(labels[idx])+1)

      img_data = preprocess.data_zoom(s1_data, s2_data)
      img_data[:,:,4:6] = np.log10(img_data[:,:,4:6])
      img_data = preprocess.data_norm(img_data, _DATASET_MEAN, _DATASET_STD)
      img_data = np.reshape(img_data,[-1]).astype(np.float32)

      example = tf.train.Example(features=tf.train.Features(feature={
          'data': _float_feature(img_data),
          'label': _int64_feature(label),
          'idx': _int64_feature(int(idx))
      }))
      writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()

def convert_dataset_balance(dataset, s1, s2, dataset_idx, per_class_num):
  sys.stdout.write('Processing ' + dataset + '\n')
  sys.stdout.flush()
  sys.stdout.write('Number of samples: %d\n' % (_NUM_CLASSES*per_class_num))
  sys.stdout.flush()

  num_per_shard = int(math.ceil(per_class_num / float(_NUM_SHARDS)))
  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        FLAGS.output_dir,
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
          s1_data = s1[idx]
          s2_data = s2[idx]
        except:
          term = i // dataset_idx[j].shape[0]
          idx = i % dataset_idx[j].shape[0]
          s1_data = s1[idx]
          s2_data = s2[idx]
          if term > 2 or term <1:
            raise RuntimeError('`term` should be 1 or 2.')
          s1_data = cv2.flip(s1_data, term-1)
          s2_data = cv2.flip(s2_data, term-1)

        label = int(j+1)
        img_data = preprocess.data_zoom(s1_data, s2_data)
        img_data[:,:,4:6] = np.log10(img_data[:,:,4:6])
        img_data = preprocess.data_norm(img_data, _DATASET_MEAN, _DATASET_STD)
        img_data = np.reshape(img_data,[-1]).astype(np.float32)

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
  s1_training, s2_training, label_training = preprocess.h5_read(path_training)
  s1_validation, s2_validation, label_validation = preprocess.h5_read(path_validation)
  num_val = label_validation.shape[0]

  class_num = np.sum(label_training, axis=0, dtype=np.uint16)
  min_class_num = np.min(class_num)
  per_class_num = min_class_num*3
  dataset_idx = []
  for i in range(_NUM_CLASSES):
    if class_num[i] < per_class_num:
      dataset_idx.append(np.where(label_training[:,i])[0])
    else:
      dataset_idx.append(np.where(label_training[:,i])[0][:per_class_num])
  
  convert_dataset_balance('train', s1_training, s2_training, dataset_idx, per_class_num)

  convert_dataset('val', s1_validation, s2_validation, label_validation, num_val)

if __name__ == '__main__':
  main()