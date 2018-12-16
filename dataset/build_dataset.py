import collections
import six
import tensorflow as tf


# A map from image format to expected data format.
_IMAGE_FORMAT_MAP = {
    'jpg': 'jpeg',
    'jpeg': 'jpeg',
    'png': 'png',
}

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self, image_format='jpeg', channels=3):
    """Class constructor.

    Args:
      image_format: Image format. Only 'jpeg', 'jpg', or 'png' are supported.
      channels: Image channels.
    """
    with tf.Graph().as_default():
      self._decode_data = tf.placeholder(dtype=tf.string)
      self._image_format = image_format
      self._session = tf.Session()
      if self._image_format in ('jpeg', 'jpg'):
        self._decode = tf.image.decode_jpeg(self._decode_data,
                                            channels=channels)
      elif self._image_format == 'png':
        self._decode = tf.image.decode_png(self._decode_data,
                                           channels=channels)

  def read_image_dims(self, image_data):
    """Reads the image dimensions.

    Args:
      image_data: string of image data.

    Returns:
      image_height and image_width.
    """
    image = self.decode_image(image_data)
    return image.shape[:2]

  def decode_image(self, image_data):
    """Decodes the image data string.

    Args:
      image_data: string of image data.

    Returns:
      Decoded image data.

    Raises:
      ValueError: Value of image channels not supported.
    """
    image = self._session.run(self._decode,
                              feed_dict={self._decode_data: image_data})
    if len(image.shape) != 3 or image.shape[2] not in (1, 3):
      raise ValueError('The image channels not supported.')

    return image


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


def protein_to_tfexample(image_data_green, image_data_red,
                         image_data_blue, image_data_yellow,
                         image_label, image_id, image_format):
  return tf.train.Example(features=tf.train.Features(feature={
            'green': _bytes_list_feature(image_data_green),
            'red': _bytes_list_feature(image_data_red),
            'blue': _bytes_list_feature(image_data_blue),
            'yellow': _bytes_list_feature(image_data_yellow),
            'label': _int64_list_feature(image_label),
            'filename': _bytes_list_feature(image_id),
            'format': _bytes_list_feature(_IMAGE_FORMAT_MAP[image_format]),}))