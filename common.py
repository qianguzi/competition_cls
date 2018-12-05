"""Provides flags that are common to scripts."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

flags = tf.app.flags

# Flags for input preprocessing.
flags.DEFINE_integer('num_classes', 18, 'Number of classes to distinguish')
# Model dependent flags.
flags.DEFINE_string('model_variant', 'xception_41', 'DeepLab model variant.')
# Defaults to None. Set multi_grid = [1, 2, 4] when using provided
# 'resnet_v1_{50,101}_beta' checkpoints.
flags.DEFINE_multi_integer('multi_grid', [1, 2, 4],
                           'Employ a hierarchy of atrous rates for ResNet.')
flags.DEFINE_float('depth_multiplier', 1.0,
                   'Multiplier for the depth (number of channels) for all '
                   'convolution ops used in MobileNet.')

FLAGS = flags.FLAGS

class ModelOptions(
    collections.namedtuple('ModelOptions', [
        'output_stride',
        'multi_grid',
        'model_variant',
        'depth_multiplier',
    ])):
  """Immutable class to hold model options."""

  __slots__ = ()

  def __new__(cls,
              output_stride=8):
    """Constructor to set default values.

    Args:
      outputs_to_num_classes: A dictionary from output type to the number of
        classes. For example, for the task of semantic segmentation with 21
        semantic classes, we would have outputs_to_num_classes['semantic'] = 21.
      crop_size: A tuple [crop_height, crop_width].
      atrous_rates: A list of atrous convolution rates for ASPP.
      output_stride: The ratio of input to output spatial resolution.

    Returns:
      A new ModelOptions instance.
    """
    return super(ModelOptions, cls).__new__(
        cls, output_stride, FLAGS.multi_grid, FLAGS.model_variant, FLAGS.depth_multiplier)

  def __deepcopy__(self, memo):
    return ModelOptions(copy.deepcopy(self.output_stride))