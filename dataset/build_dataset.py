import sys
import os, cv2
import math, h5py
import numpy as np
import tensorflow as tf

from dataset.preprocess import first

flags = tf.app.flags

flags.DEFINE_string('dataset_folder', '/home/data/lcz', 'Folder containing dataset.')
flags.DEFINE_string('output_dir', '/media/jun/data/lcz/tfrecord', 'Output location of dataset.')
flags.DEFINE_string('preprocess_method', 'default', 'The image data preprocess term.')
flags.DEFINE_integer('aug_factor', '3', 'The factor of data augmetation.')
FLAGS = flags.FLAGS

_NUM_CLASSES = 17
_NUM_SHARDS = 4
PREPROCESS_METHOD = {
    'default': first.img_data_preprocess,
    'first': first.img_data_preprocess,
}

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def data_aug(s1_data, s2_data, method):
  if method == 2 or method == 1:
    s1_data = cv2.flip(s1_data, method-1)
    s2_data = cv2.flip(s2_data, method-1)
  elif method == 3:
    s1_data = cv2.flip(s1_data, -1)
    s2_data = cv2.flip(s2_data, -1)
  else:
    raise RuntimeWarning('The augmentation factor is not supported yet.')
  return s1_data, s2_data 

def convert_dataset(dataset, s1, s2, labels, num_samples, preprocess_fn):
  sys.stdout.write('Processing ' + dataset + '\n')
  sys.stdout.flush()
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

      example = tf.train.Example(features=tf.train.Features(feature={
          'data': _float_feature(img_data),
          'label': _int64_feature(label),
          'idx': _int64_feature(int(idx))
      }))
      writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()

def convert_dataset_balance(dataset, s1, s2, dataset_idx, per_class_num, preprocess_fn):
  sys.stdout.write('Processing ' + dataset + '\n')
  sys.stdout.flush()
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
          s1_data = s1[idx]
          s2_data = s2[idx]
        except:
          method = i // dataset_idx[j].shape[0]
          if method <= 0:
             raise RuntimeError('The augmentation method must be positive.')
          idx = i % dataset_idx[j].shape[0]
          s1_data = s1[idx]
          s2_data = s2[idx]
          data_aug(s1_data, s2_data, method)

        label = int(j+1)
        img_data = preprocess_fn(s1_data, s2_data)

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
  num_val = int(label_validation.shape[0])

  class_num = np.sum(label_training, axis=0, dtype=np.uint16)
  min_class_num = np.min(class_num)
  per_class_num = min_class_num * FLAGS.aug_factor
  dataset_idx = []
  for i in range(_NUM_CLASSES):
    if class_num[i] < per_class_num:
      dataset_idx.append(np.where(label_training[:,i])[0])
    else:
      dataset_idx.append(np.where(label_training[:,i])[0][:per_class_num])
  
  if FLAGS.preprocess_method not in PREPROCESS_METHOD:
    raise ValueError('The specified preprocess method is not supported yet.')
  preprocess_fn = PREPROCESS_METHOD[FLAGS.preprocess_method]
  tf.gfile.MakeDirs(FLAGS.output_dir)
  tf.gfile.MakeDirs(os.path.join(FLAGS.output_dir, FLAGS.preprocess_method))

  convert_dataset_balance('train', s1_training, s2_training, dataset_idx, per_class_num, preprocess_fn)

  convert_dataset('val', s1_validation, s2_validation, label_validation, num_val, preprocess_fn)

if __name__ == '__main__':
  main()