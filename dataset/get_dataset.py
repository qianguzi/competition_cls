import os
import collections
import h5py, glob
import tensorflow as tf

# Named tuple to describe the dataset properties.
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    ['channel',
     'splits_to_sizes',
    ]
)

_DEFAULT_INFORMATION = DatasetDescriptor(
    channel=18,
    splits_to_sizes={
        'train': 121992,
        'val': 24119,
    },
)

_FIRST_INFORMATION = DatasetDescriptor(
    channel=18,
    splits_to_sizes={
        'train': 121992,
        'val': 24119,
    },
)

_DATASETS_INFORMATION = {
    'default': _DEFAULT_INFORMATION,
    'first': _FIRST_INFORMATION,
}

# Default file pattern of TFRecord of TensorFlow Example.
_FILE_PATTERN = '%s-*'

def get_dataset(dataset_name, split_name, dataset_dir, batch_size, is_training=True):
  '''Get batch of dataset.'''
  if dataset_name not in _DATASETS_INFORMATION:
    raise ValueError('The specified dataset is not supported yet.')
  
  splits_to_sizes = _DATASETS_INFORMATION[dataset_name].splits_to_sizes
  if split_name not in splits_to_sizes:
    raise ValueError('Data split name %s not recognized.' % split_name)
  num_samples = splits_to_sizes[split_name]

  file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, dataset_name, file_pattern % split_name)
  file_path = glob.glob(file_pattern)
  filename_queue = tf.train.string_input_producer(
      file_path, shuffle=is_training)

  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  channel = _DATASETS_INFORMATION[dataset_name].channel
  features = tf.parse_single_example(
      serialized_example,
      features={'data': tf.FixedLenFeature([32, 32, channel], tf.float32),
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
  return samples, num_samples