import os.path
import tensorflow as tf
from tensorflow.contrib import slim
from glob import glob

from dataset import data_augmentation
from dataset import dataset_information

tfexample_decoder = slim.tfexample_decoder
dataset_data_provider = slim.dataset_data_provider
DATASETS_INFORMATION = dataset_information.DATASETS_INFORMATION

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A image of multi-channels.',
    'labels': 'class',
}
# Default file pattern of TFRecord of TensorFlow Example.
_FILE_PATTERN = '%s-*'


def get_batch(dataset, image_size, batch_size,
              channel=0, num_readers=1, num_threads=1,
              is_training=True, scope=None):
  """Gets an instance of slim Dataset.

  Args:
    dataset: Dataset.
    image_size: Image size [height, width].
    batch_size: Batch size.
    num_readers: Number of readers for data provider.
    num_threads: Number of threads for batching data.
    is_training: Is training or not.
    scope: The name of op.

  Returns:
    A dictionary of batched Tensors for classification.
  """
  with tf.name_scope(scope, 'Dataset_quene'):
    data_provider = dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=num_readers,
        num_epochs=None if is_training else 1,
        shuffle=is_training)
    image, label, image_name = DATASETS_INFORMATION[dataset.name].get_data_fn(data_provider)
    if channel:
      image = image[:,:,:channel]
    image = data_augmentation.preprocess_image(
        image,
        height=image_size[0],
        width=image_size[1],
        is_training=is_training)
    sample = {
        'image': image,
        'label': label,
        'image_name': image_name,
        }
    return tf.train.batch(
               sample,
               batch_size=batch_size,
               num_threads=num_threads,
               capacity=5*batch_size,
               allow_smaller_final_batch=not is_training,
               dynamic_pad=True)


def get_dataset(dataset_name, dataset_dir, split_name, 
                is_training=True, scope=None, **kwargs):
  """Gets an instance of slim Dataset.

  Args:
    dataset_name: Dataset name.
    dataset_dir: The directory of the dataset sources.
    split_name: A Split name.
    is_training: Is training or not.
    scope: The name of op.

  Returns:
    A dictionary of batched Tensors for classification.

  Raises:
    ValueError: if the dataset_name or split_name is not recognized.
  """
  if dataset_name not in DATASETS_INFORMATION:
    raise ValueError('The specified dataset is not supported yet.')

  splits_to_sizes = DATASETS_INFORMATION[dataset_name].splits_to_sizes

  if split_name not in splits_to_sizes:
    raise ValueError('data split name %s not recognized' % split_name)

  file_pattern = _FILE_PATTERN
  if is_training:
    num_samples = DATASETS_INFORMATION[dataset_name].total_samples - splits_to_sizes[split_name]
  else:
    num_samples = splits_to_sizes[split_name]

  # Specify how the TF-Examples are decoded.
  keys_to_features = DATASETS_INFORMATION[dataset_name].keys_to_features
  items_to_handlers = DATASETS_INFORMATION[dataset_name].items_to_handlers

  decoder = tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  with tf.name_scope(scope, 'Dataset_quene'):
    image_class_list = []
    label_class_list = []
    name_class_list = []
    idx_to_name = DATASETS_INFORMATION[dataset_name].idx_to_name
    for class_name in idx_to_name.values():
      class_dir = os.path.join(dataset_dir, dataset_name, class_name)
      if is_training:
        files = glob(os.path.join(class_dir, file_pattern % dataset_name))
        files.remove(glob(os.path.join(class_dir, file_pattern % split_name))[0])
      else:
        files = glob(os.path.join(class_dir, file_pattern % split_name))
      dataset = slim.dataset.Dataset(
                    data_sources=files,
                    reader=tf.TFRecordReader,
                    decoder=decoder,
                    num_samples=num_samples,
                    items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
                    name=dataset_name)
      sample = get_batch(dataset, is_training=is_training, 
                         scope='Dataset_quene_%s'%(class_name), **kwargs)
      image_class_list.append(sample['image'])
      label_class_list.append(sample['label'])
      name_class_list.append(sample['image_name'])
    image = tf.concat(image_class_list, 0)
    label = tf.concat(label_class_list, 0)
    image_name = tf.concat(name_class_list, 0)
    samples = {
        'image': image,
        'label': label,
        'image_name': image_name,
        }
    return samples, num_samples


if __name__ == '__main__':
  tf_dir = '/media/jun/data/lcz/tfrecord'
  samples, _ = get_dataset('lcz', tf_dir, 'lcz-05', image_size=[64,64], batch_size=4, channel=1)
  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())

  with tf.Session() as sess:
      sess.run(init_op)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      try:
          while not coord.should_stop():
              s = sess.run(samples)
              print(s['image_name'])
      except tf.errors.OutOfRangeError:
          coord.request_stop()
          coord.join(threads)
