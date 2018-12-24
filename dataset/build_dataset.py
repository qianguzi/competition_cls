import os, sys
import math, h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from random import shuffle

import dataset_information

flags = tf.app.flags
# /media/deeplearning/f3cff4c9-1ab9-47f0-8b82-231dedcbd61b/lcz
# /mnt/home/hdd/hdd1/home/LiaoL/Kaggle/Protein/dataset
# /mnt/home/hdd/hdd1/home/junq/dataset
flags.DEFINE_enum('dataset_name', 'protein', ['protein', 'lcz'], 'Dataset name.')
flags.DEFINE_string('dataset_folder', '/media/jun/data/protein',
                    'Folder containing dataset_name.')
flags.DEFINE_float('split_factor', 0.99, 'The image data preprocess term.')
flags.DEFINE_string('output_folder', '/media/jun/data/tfrecord',
                    'Folder containing dataset_name.')
FLAGS = flags.FLAGS


def convert_tfrecord_class(dataset_info, ori_data, per_class_image_ids, per_class_counts, num_shards=6):
  dataset_dir = FLAGS.dataset_folder
  output_dir = os.path.join(FLAGS.output_folder, dataset_info.dataset_name)
  tf.gfile.MakeDirs(output_dir)
  num_samples = 0
  split_class_list = []
  for label_idx in range(dataset_info.num_classes):
    label_name = dataset_info.idx_to_name[label_idx]
    tf.gfile.MakeDirs(os.path.join(output_dir, label_name))
    image_ids = per_class_image_ids[label_name]
    class_counts = per_class_counts[label_name]
    num_samples += class_counts
    sys.stdout.write('[%d] Processing %s, number: %d\n' % (label_idx, label_name, class_counts))
    sys.stdout.flush()
    split_size_list = []
    if FLAGS.split_factor > 0:
      num_per_shard = int(math.floor(class_counts*FLAGS.split_factor))
      num_shards = 2
    else:
      num_per_shard = int(math.ceil(class_counts/float(num_shards)))
    for shard_id in range(num_shards):
      output_filename = os.path.join(
          output_dir, label_name, 
          '%s-%02d-of-%02d.tfrecord' % (dataset_info.dataset_name, shard_id+1, num_shards))
      writer = tf.python_io.TFRecordWriter(output_filename)
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id+1)*num_per_shard, class_counts)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i+1, class_counts, shard_id+1))
        sys.stdout.flush()
        image_id = image_ids[i]
        example = dataset_info.data_to_tfexample_fn(
            dataset_info, ori_data, image_id, label_idx, dataset_dir)
        writer.write(example.SerializeToString())
      split_size = end_idx-start_idx
      split_size_list.append(split_size)
      sys.stdout.write('\n--- Split size: %d ---\n'%(split_size))
      sys.stdout.flush()
    split_class_list.append(split_size_list)
  total_split_size = np.sum(split_class_list, 0)
  print('[*] Total split size: ', total_split_size)
  sys.stdout.write('[*] Number of total samples: %d\n' % (num_samples))
  sys.stdout.flush()


def build_protein_dataset():
  train_data = pd.read_csv(os.path.join(FLAGS.dataset_folder, 'train.csv'))
  dataset_info = dataset_information.DATASETS_INFORMATION['protein']
  for key in dataset_info.idx_to_name.keys():
    train_data[dataset_info.idx_to_name[key]] = 0
  def fill_targets(row):
    row.Target = np.array(row.Target.split(' ')).astype(np.int)
    for num in row.Target:
      name = dataset_info.idx_to_name[int(num)]
      row.loc[name] = 1
    return row
  train_data = train_data.apply(fill_targets, axis=1)
  train_data['Number_of_targets'] = train_data.drop(['Id', 'Target'],axis=1).sum(axis=1)

  per_class_counts = {}
  per_class_image_ids = {}
  sub_data = train_data.drop(['Target'],axis=1).copy(deep=True)
  target_counts = sub_data.drop(['Id'],axis=1).sum(axis=0).sort_values(ascending=False)
  for i in range(dataset_info.num_classes):
    special_target = target_counts.keys()[-1]
    per_class_counts[special_target] = target_counts[-1]
    special_target_id = list(sub_data[sub_data[special_target] == 1]['Id'])
    shuffle(special_target_id)
    per_class_image_ids[special_target] = special_target_id
    sub_data = sub_data[sub_data[special_target] == 0].drop([special_target],axis=1)
    target_counts = sub_data.drop(['Id'],axis=1).sum(axis=0).sort_values(ascending=False)

  convert_tfrecord_class(dataset_info, train_data, per_class_image_ids, per_class_counts)


def build_lcz_dataset():
  dataset_info = dataset_information.DATASETS_INFORMATION['lcz']
  path_training = os.path.join(FLAGS.dataset_folder, 'lcz', 'training.h5')
  path_validation = os.path.join(FLAGS.dataset_folder, 'lcz', 'validation.h5')
  fid_training = h5py.File(path_training,'r')
  s1_training = fid_training['sen1']
  s2_training = fid_training['sen2']
  label_training = fid_training['label']
  fid_validation = h5py.File(path_validation,'r')
  s1_validation = fid_validation['sen1']
  s2_validation = fid_validation['sen2']
  label_validation = fid_validation['label']
  s1 = {'training': s1_training, 'validation':s1_validation}
  s2 = {'training': s2_training, 'validation':s2_validation}
  label = {'training': label_training, 'validation':label_validation}

  def split_dataset(labels, sub_dataset_name='default'):
    counts = {}
    image_ids = {}
    for i in range(dataset_info.num_classes):
      ids = np.where(labels[:,i])[0]
      image_ids[dataset_info.idx_to_name[i]] = [(sub_dataset_name, idx) for idx in ids]
      counts[dataset_info.idx_to_name[i]] = len(ids)
    return image_ids, counts

  train_image_ids, train_counts = split_dataset(label_training, 'training')
  val_image_ids, val_counts = split_dataset(label_validation, 'validation')

  per_class_image_ids ={}
  per_class_counts = {}
  for class_name in train_image_ids:
    class_ids = train_image_ids[class_name] + val_image_ids[class_name]
    shuffle(class_ids)
    per_class_image_ids[class_name] = class_ids
    per_class_counts[class_name] = train_counts[class_name] + val_counts[class_name]

  convert_tfrecord_class(dataset_info, [s1, s2, label], per_class_image_ids, per_class_counts, 20)


_BUILD_DATASET_FN={
  'protein': build_protein_dataset,
  'lcz': build_lcz_dataset,
}

def main(unused_arg):
  _BUILD_DATASET_FN[FLAGS.dataset_name]()


if __name__ == '__main__':
  tf.app.run(main)
