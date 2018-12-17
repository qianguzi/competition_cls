import collections
import os, six
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from dataset.preprocess import default
tfexample_decoder = slim.tfexample_decoder

# A map from image format to expected data format.
_IMAGE_FORMAT_MAP = {
    'jpg': 'jpeg',
    'jpeg': 'jpeg',
    'png': 'png',
}

_PROTEIN_CLASS_NAMES = {
  0: 'Nucleoplasm',  
  1: 'Nuclear_membrane',   
  2: 'Nucleoli',   
  3: 'Nucleoli_fibrillar_center',   
  4: 'Nuclear_speckles',
  5: 'Nuclear_bodies',   
  6: 'Endoplasmic_reticulum',   
  7: 'Golgi_apparatus',   
  8: 'Peroxisomes',   
  9: 'Endosomes',   
  10: 'Lysosomes',   
  11: 'Intermediate_filaments',   
  12: 'Actin_filaments',   
  13: 'Focal_adhesion_sites',   
  14: 'Microtubules',   
  15: 'Microtubule_ends',   
  16: 'Cytokinetic_bridge',   
  17: 'Mitotic_spindle',   
  18: 'Microtubule_organizing_center',   
  19: 'Centrosome',   
  20: 'Lipid_droplets',   
  21: 'Plasma_membrane',   
  22: 'Cell_junctions',   
  23: 'Mitochondria',   
  24: 'Aggresome',   
  25: 'Cytosol',   
  26: 'Cytoplasmic_bodies',   
  27: 'Rods_and_rings'
}

_LCZ_CLASS_NAMES = {
  0: 'Compact_highrise',  
  1: 'Compact_midrise',   
  2: 'Compact_lowrise',   
  3: 'Open_highrise',   
  4: 'Open_midrise',
  5: 'Open_lowrise',   
  6: 'Lightweight_lowrise',   
  7: 'Large_lowrise',   
  8: 'Sparsely_built',   
  9: 'Heavy_industry',   
  10: 'Dense_trees',   
  11: 'Scattered_trees',   
  12: 'Bush_and_scrub',   
  13: 'Low_plants',   
  14: 'Bare_rock_or_paved',   
  15: 'Bare_soil_or_sand',   
  16: 'Water', 
}


def _int64_list_feature(values):
  """Returns a TF-Feature of int64_list.

  Args:
    values: A scalar or list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, collections.Iterable):
    values = [values]

  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float_list_feature(values):
  """Returns a TF-Feature of float_list.

  Args:
    values: A scalar or list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, collections.Iterable):
    values = [values]

  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _bytes_list_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  def norm2bytes(value):
    return value.encode() if isinstance(value, str) and six.PY3 else value

  return tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))


def _lcz_to_tfexample(dataset_info, ori_data, image_id, label_idx, dataset_folder=None):
  name, idx = image_id
  label = ori_data[2][name][idx]
  label = np.array(label, np.float32)
  if label[label_idx] != 1.:
    raise RuntimeError('Label is wrong.')
  s1_data = ori_data[0][name][idx]
  s2_data = ori_data[1][name][idx]
  img_data = default.new_preprocess(s1_data, s2_data)
  img_data = np.reshape(img_data, [-1]).astype(np.float32)
  image_name = (name+str(idx)).encode()
  return tf.train.Example(features=tf.train.Features(feature={
             'data': _float_list_feature(img_data),
             'label': _float_list_feature(label),
             'filename': _bytes_list_feature(image_name),
             }))

def _get_lcz_data(data_provider):
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
  image, label = data_provider.get(['data', 'label'])
  # Some datasets do not contain image_name.
  if 'image_name' in data_provider.list_items():
    image_name, = data_provider.get(['image_name'])
  else:
    image_name = tf.constant('')
  return image, label, image_name


def _protein_to_tfexample(dataset_info, ori_data, image_id, label_idx, dataset_folder=None):
  label = list(ori_data[ori_data['Id']==image_id]['Target'])[0]
  if label_idx not in label:
    raise RuntimeError('Label is wrong.')
  one_hot_label = np.zeros([dataset_info.num_classes], np.float32)
  for l in label:
    one_hot_label[l] = 1.
  image_filename = os.path.join(dataset_folder, 'train', image_id)
  image_data_green = tf.gfile.FastGFile(image_filename+'_green'+'.png', 'rb').read()
  image_data_red = tf.gfile.FastGFile(image_filename+'_red'+'.png', 'rb').read()
  image_data_blue = tf.gfile.FastGFile(image_filename+'_blue'+'.png', 'rb').read()
  image_data_yellow = tf.gfile.FastGFile(image_filename+'_yellow'+'.png', 'rb').read()
  return tf.train.Example(features=tf.train.Features(feature={
             'green': _bytes_list_feature(image_data_green),
             'red': _bytes_list_feature(image_data_red),
             'blue': _bytes_list_feature(image_data_blue),
             'yellow': _bytes_list_feature(image_data_yellow),
             'label': _float_list_feature(one_hot_label),
             'filename': _bytes_list_feature(image_id),
             'format': _bytes_list_feature(_IMAGE_FORMAT_MAP['png']),
             }))

def _get_protein_data(data_provider):
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
  label, = data_provider.get(['label'])
  image = data_provider.get(
      ['image_green', 'image_red', 'image_blue', 'image_yellow'])
  image = tf.concat(image, -1)
  image = tf.cast(image / 255, tf.float32)
  # Some datasets do not contain image_name.
  if 'image_name' in data_provider.list_items():
    image_name, = data_provider.get(['image_name'])
  else:
    image_name = tf.constant('')
  return image, label, image_name

# Named tuple to describe the dataset properties.
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    [
      'dataset_name',
      'splits_to_sizes',   # Splits of the dataset into training, val, and test.
      'total_samples',
      'num_classes',
      'idx_to_name',
      'data_to_tfexample_fn',
      'keys_to_features',
      'items_to_handlers',
      'get_data_fn',
    ]
)

_PROTEIN_INFORMATION = DatasetDescriptor(
    dataset_name='protein',
    splits_to_sizes={
        'protein-01': 5188,
        'protein-02': 5188,
        'protein-03': 5188,
        'protein-04': 5188,
        'protein-05': 5188,
        'protein-06': 5132,
    },
    total_samples=31072,
    num_classes=28,
    idx_to_name=_PROTEIN_CLASS_NAMES,
    data_to_tfexample_fn=_protein_to_tfexample,
    keys_to_features={
      'green': tf.FixedLenFeature((), tf.string, default_value=''),
      'red': tf.FixedLenFeature((), tf.string, default_value=''),
      'blue': tf.FixedLenFeature((), tf.string, default_value=''),
      'yellow': tf.FixedLenFeature((), tf.string, default_value=''),
      'label': tf.FixedLenFeature([28], tf.float32),
      'filename': tf.FixedLenFeature((), tf.string, default_value=''),
      'format': tf.FixedLenFeature((), tf.string, default_value='png'),
      },
    items_to_handlers={
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
      },
    get_data_fn=_get_protein_data,
)

_LCZ_INFORMATION = DatasetDescriptor(
    dataset_name='lcz',
    splits_to_sizes={
        'lcz-01': 18833,
        'lcz-02': 18833,
        'lcz-03': 18833,
        'lcz-04': 18833,
        'lcz-05': 18833,
        'lcz-06': 18833,
        'lcz-07': 18833,
        'lcz-08': 18833,
        'lcz-09': 18833,
        'lcz-10': 18833,
        'lcz-11': 18833,
        'lcz-12': 18833,
        'lcz-13': 18833,
        'lcz-14': 18833,
        'lcz-15': 18833, 
        'lcz-16': 18833,
        'lcz-17': 18833,
        'lcz-18': 18833,
        'lcz-19': 18833,
        'lcz-20': 18658,
    },
    total_samples=376485,
    num_classes=17,
    idx_to_name=_LCZ_CLASS_NAMES,
    data_to_tfexample_fn=_lcz_to_tfexample,
    keys_to_features={
      'data': tf.FixedLenFeature([32, 32, 7], tf.float32),
      'label': tf.FixedLenFeature([17], tf.float32),
      'filename': tf.FixedLenFeature((), tf.string, default_value=''),
      },
    items_to_handlers={
      'data': tfexample_decoder.Tensor('data', shape=[32, 32, 7]),
      'label': tfexample_decoder.Tensor('label', shape=[17]),
      'image_name': tfexample_decoder.Tensor('filename'),
      },
    get_data_fn=_get_lcz_data,
)

DATASETS_INFORMATION = {
    'protein': _PROTEIN_INFORMATION,
    'lcz': _LCZ_INFORMATION,
}
