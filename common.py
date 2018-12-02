"""Provides flags that are common to scripts."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.app.flags

# Flags for input preprocessing.
flags.DEFINE_integer('num_classes', 18, 'Number of classes to distinguish')
flags.DEFINE_string('model_variant', 'mobilenet', 'DeepLab model variant.')
flags.DEFINE_float('depth_multiplier', 1.4,
                   'Multiplier for the depth (number of channels) for all '
                   'convolution ops used in MobileNet.')

FLAGS = flags.FLAGS