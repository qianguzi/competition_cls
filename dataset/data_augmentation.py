"""Provides utilities to preprocess images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.image import rotate # pylint: disable=E0611
from tensorflow.python.ops import control_flow_ops # pylint: disable=E0611


def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, scope=None):
  """Distort the color of a Tensor image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1].
  Raises:
    ValueError: if color_ordering not in [0, 1].
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.75, upper=1.5)
    elif color_ordering == 1:
      image = tf.image.random_contrast(image, lower=0.75, upper=1.5)
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
        raise ValueError('color_ordering must be in [0, 1]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_for_train(image, height, width, scope=None):
  """Distort one image for training a network.

  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.

  Additionally it would create image_summaries to display the different
  transformations applied to the image.

  Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [-1, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
    height: integer
    width: integer
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of distorted image used for training with range [-1, 1].
  """
  with tf.name_scope(scope, 'distort_image', [image, height, width]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Randomly select resize method.
    distorted_image = apply_with_random_selector(
        image,
        lambda x, method: tf.image.resize_images(x, [height, width], method),
        num_cases=4)

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # Randomly rotate the image.
    random_angles = tf.random_uniform([], minval=-np.pi, maxval=np.pi, dtype=tf.float32)
    distorted_image = rotate(distorted_image, random_angles)
    # Randomly distort the colors.
    distorted_image = apply_with_random_selector(
        distorted_image,
        lambda x, ordering: distort_color(x, ordering),
        num_cases=2)

    # Randomly distort the colors.
    distorted_image = tf.subtract(distorted_image, 0.5)
    distorted_image = tf.multiply(distorted_image, 2.0)
    return distorted_image


def preprocess_for_eval(image, height, width, scope=None):
  """Prepare one image for evaluation.

  If height and width are specified it would output an image with that size by
  applying resize_bilinear.

  Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
    height: integer
    width: integer
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  with tf.name_scope(scope, 'eval_image', [image, height, width]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    if height and width:
      # Resize the image to the specified height and width.
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, [height, width],
                                       align_corners=False)
      image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def preprocess_image(image, height, width, is_training=False):
  """Pre-process one image for training or evaluation.

  Args:
    image: 3-D Tensor [height, width, channels] with the image. If dtype is
      tf.float32 then the range should be [0, 1], otherwise it would converted
      to tf.float32 assuming that the range is [0, MAX], where MAX is largest
      positive representable number for int(8/16/32) data type (see
      `tf.image.convert_image_dtype` for details).
    height: integer, image expected height.
    width: integer, image expected width.
    is_training: Boolean. If true it would transform an image for train,
      otherwise it would transform it for evaluation.

  Returns:
    3-D float Tensor containing an appropriately scaled image

  Raises:
    ValueError: if user does not provide bounding box
  """
  if is_training:
    return preprocess_for_train(image, height, width)
  else:
    return preprocess_for_eval(image, height, width)
