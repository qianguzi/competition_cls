"""Provides flags that are common to scripts."""
import tensorflow as tf

flags = tf.app.flags

# Flags for input preprocessing.
flags.DEFINE_integer('num_classes', 18, 'Number of classes to distinguish')
flags.DEFINE_string('model_variant', 'mobilenet', 'DeepLab model variant.')
flags.DEFINE_float('depth_multiplier', 1.5,
                   'Multiplier for the depth (number of channels) for all '
                   'convolution ops used in MobileNet.')

FLAGS = flags.FLAGS