import collections
import os.path
import tensorflow as tf
from tensorflow.contrib import slim

import data_augmentation

dataset = slim.dataset

tfexample_decoder = slim.tfexample_decoder
dataset_data_provider = slim.dataset_data_provider

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'labels_class': ('A semantic segmentation label whose size matches image.'
                     'Its values range from 0 (background) to num_classes.'),
}

# Named tuple to describe the dataset properties.
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    ['splits_to_sizes',   # Splits of the dataset into training, val, and test.
     'total_samples',
     'num_classes',
    ]
)

_PROTEIN_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'protein-01': 2975,
        'protein-02': 2975,
        'protein-03': 2975,
        'protein-04': 2975,
        'protein-05': 2975,
        'protein-06': 2975,
    },
    total_samples=30127,
    num_classes=28,
)

_LCZ_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 20210,  # num of samples in images/training
        'val': 2000,  # num of samples in images/validation
    },
    total_samples=30127,
    num_classes=17,
)

_DATASETS_INFORMATION = {
    'protein': _PROTEIN_INFORMATION,
    'lcz': _LCZ_INFORMATION,
}

# Default file pattern of TFRecord of TensorFlow Example.
_FILE_PATTERN = '%s-*'


def get_dataset(dataset_name, split_name, dataset_dir):
  """Gets an instance of slim Dataset.

  Args:
    dataset_name: Dataset name.
    split_name: A Split name.
    dataset_dir: The directory of the dataset sources.

  Returns:
    An instance of slim Dataset.

  Raises:
    ValueError: if the dataset_name or split_name is not recognized.
  """
  if dataset_name not in _DATASETS_INFORMATION:
    raise ValueError('The specified dataset is not supported yet.')

  splits_to_sizes = _DATASETS_INFORMATION[dataset_name].splits_to_sizes

  if split_name not in splits_to_sizes:
    raise ValueError('data split name %s not recognized' % split_name)

  # Prepare the variables for different datasets.
  num_classes = _DATASETS_INFORMATION[dataset_name].num_classes
  num_samples = _DATASETS_INFORMATION[dataset_name].total_samples - splits_to_sizes[split_name]

  file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Specify how the TF-Examples are decoded.
  keys_to_features = {
      'green': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'red': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'blue': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'yellow': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'label': tf.FixedLenFeature(
          [28], tf.int64, default_value=''),
      'filename': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'format': tf.FixedLenFeature(
          (), tf.string, default_value='png'),
  }
  items_to_handlers = {
      'image_green': tfexample_decoder.Image(
          image_key='green',
          format_key='format',
          channels=1),
      'image_red': tfexample_decoder.Image(
          image_key='red',
          format_key='format',
          channels=1),
      'image_blue': tfexample_decoder.Image(
          image_key='blue',
          format_key='format',
          channels=1),
      'image_yellow': tfexample_decoder.Image(
          image_key='yellow',
          format_key='format',
          channels=1),
      'label': tfexample_decoder.Tensor('label', shape=[28]),
      'image_name': tfexample_decoder.Tensor('filename'),
  }

  decoder = tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  return dataset.Dataset(
      data_sources=file_pattern,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=splits_to_sizes[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=num_classes,
      name=dataset_name,
      multi_label=True)


def _get_data(data_provider):
  """Gets data from data provider.

  Args:
    data_provider: An object of slim.data_provider.

  Returns:
    image: Image Tensor.
    label: Label Tensor storing segmentation annotations.
    image_name: Image name.

  Raises:
    ValueError: Failed to find label.
  """
  if 'label' not in data_provider.list_items():
    raise ValueError('Failed to find labels.')
  #img_green, img_red, img_blue, img_yellow
  image = data_provider.get(
      ['image_green', 'image_red', 'image_blue', 'image_yellow'])

  # Some datasets do not contain image_name.
  if 'image_name' in data_provider.list_items():
    image_name, = data_provider.get(['image_name'])
  else:
    image_name = tf.constant('')

  label, = data_provider.get(['label'])

  return image, label, image_name


def get(dataset,
        crop_size,
        batch_size,
        num_readers=1,
        num_threads=4,
        is_training=True):
  """Gets the dataset split for semantic segmentation.

  This functions gets the dataset split for semantic segmentation. In
  particular, it is a wrapper of (1) dataset_data_provider which returns the raw
  dataset split, (2) input_preprcess which preprocess the raw data, and (3) the
  Tensorflow operation of batching the preprocessed data. Then, the output could
  be directly used by training, evaluation or visualization.

  Args:
    dataset: An instance of slim Dataset.
    crop_size: Image crop size [height, width].
    batch_size: Batch size.
    num_readers: Number of readers for data provider.
    num_threads: Number of threads for batching data.
    is_training: Is training or not.

  Returns:
    A dictionary of batched Tensors for semantic segmentation.
  """
  data_provider = dataset_data_provider.DatasetDataProvider(
      dataset,
      num_readers=num_readers,
      num_epochs=None if is_training else 1,
      shuffle=is_training)
  image, label, image_name = _get_data(data_provider)
  image = tf.stack(image, -1)
  image= data_augmentation.preprocess_image(
      image,
      height=crop_size[0],
      width=crop_size[1],
      is_training=is_training)
  sample = {
      'image': image,
      'image_name': image_name,
  }
  if label is not None:
    sample['label'] = label

  return tf.train.batch(
      sample,
      batch_size=batch_size,
      num_threads=num_threads,
      capacity=32 * batch_size,
      allow_smaller_final_batch=not is_training,
      dynamic_pad=True)