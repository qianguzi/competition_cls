import h5py, glob
import tensorflow as tf

import common


def get_dataset(file_path, batch_size, is_training=True):
  '''Get batch of dataset.'''
  reader = tf.TFRecordReader()

  files_path = glob.glob(file_path)
  filename_queue = tf.train.string_input_producer(
      files_path, shuffle=is_training)

  _, serialized_example = reader.read(filename_queue)

  features = tf.parse_single_example(
      serialized_example,
      features={'data': tf.FixedLenFeature([32, 32, common.channel], tf.float32),
                'label': tf.FixedLenFeature([], tf.int64),
                'idx': tf.FixedLenFeature([], tf.int64)})
  sample = {
      'data': features['data'],
      'label': features['label'],
      'idx': features['idx']
  }

  samples = tf.train.batch(sample,
                           batch_size, 
                           num_threads=4,
                           capacity=5*batch_size,
                           allow_smaller_final_batch=not is_training)
  return samples