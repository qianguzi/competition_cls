"""Provides flags that are common to scripts."""
import tensorflow as tf

from dataset.preprocess import first

flags = tf.app.flags

# Flags for input preprocessing.
flags.DEFINE_integer('num_classes', 18, 'Number of classes to distinguish')
flags.DEFINE_string('preprocess_term', 'default', 'The image data preprocess term.')
flags.DEFINE_float('depth_multiplier', 1.5,
                   'Multiplier for the depth (number of channels) for all '
                   'convolution ops used in MobileNet.')
flags.DEFINE_string('tfrecord_dir', '/media/jun/data/lcz/tfrecord', 'Location of dataset')

FLAGS = flags.FLAGS

_PREPROCESS_TERM={
  'default': first.img_data_preprocess,
  'first': first.img_data_preprocess,
}

_PREPROCESS_CHANNEL={
  'default': 7,
  'first': 18,
}

try:
  preprocess_fn = _PREPROCESS_TERM[FLAGS.preprocess_term]
  channel = _PREPROCESS_CHANNEL[FLAGS.preprocess_term]
except:
  print('No this %s prrprocess term, `default` term will be selected.'%(FLAGS.preprocess_term))
  preprocess_fn = _PREPROCESS_TERM['defalut']
  channel = _PREPROCESS_CHANNEL['default']