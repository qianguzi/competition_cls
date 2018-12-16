import os, sys, math
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

#from dataset import build_dataset
import build_dataset

flags = tf.app.flags
#/media/deeplearning/f3cff4c9-1ab9-47f0-8b82-231dedcbd61b/lcz
flags.DEFINE_string('dataset_folder', '/media/jun/data/protein', 'Folder containing dataset.')
flags.DEFINE_string('output_dir', '/media/jun/data/protein/tfrecord', 'Output location of dataset.')
tf.app.flags.DEFINE_enum('image_format', 'png', ['jpg', 'jpeg', 'png'], 'Image format.')
FLAGS = flags.FLAGS

_NUM_CLASSES = 28
_IDX_TO_NAME = {
  0: 'Nucleoplasm',  
  1: 'Nuclear membrane',   
  2: 'Nucleoli',   
  3: 'Nucleoli fibrillar center',   
  4: 'Nuclear speckles',
  5: 'Nuclear bodies',   
  6: 'Endoplasmic reticulum',   
  7: 'Golgi apparatus',   
  8: 'Peroxisomes',   
  9: 'Endosomes',   
  10: 'Lysosomes',   
  11: 'Intermediate filaments',   
  12: 'Actin filaments',   
  13: 'Focal adhesion sites',   
  14: 'Microtubules',   
  15: 'Microtubule ends',   
  16: 'Cytokinetic bridge',   
  17: 'Mitotic spindle',   
  18: 'Microtubule organizing center',   
  19: 'Centrosome',   
  20: 'Lipid droplets',   
  21: 'Plasma membrane',   
  22: 'Cell junctions',   
  23: 'Mitochondria',   
  24: 'Aggresome',   
  25: 'Cytosol',   
  26: 'Cytoplasmic bodies',   
  27: 'Rods & rings'
}
_NAME_TO_IDX = dict((v,k) for k,v in _IDX_TO_NAME.items())
_NUM_SHARDS = 11


def fill_targets(row):
    row.Target = np.array(row.Target.split(' ')).astype(np.int)
    for num in row.Target:
        name = _IDX_TO_NAME[int(num)]
        row.loc[name] = 1
    return row


def convert_tfrecord(dataset, ori_data, per_class_image_ids, per_class_counts):
  tf.gfile.MakeDirs(FLAGS.output_dir)
  num_samples = 0
  for label_idx in range(_NUM_CLASSES):
    label_name = _IDX_TO_NAME[label_idx]
    tf.gfile.MakeDirs(os.path.join(FLAGS.output_dir, label_name))
    image_ids = per_class_image_ids[label_name]
    class_counts = per_class_counts[label_name]
    num_samples += class_counts
    sys.stdout.write('[*] Processing %s, number: %d\n' % (label_name, class_counts))
    sys.stdout.flush()

    num_per_shard = int(math.ceil(class_counts/float(_NUM_SHARDS)))
    for shard_id in range(_NUM_SHARDS):
      output_filename = os.path.join(
          FLAGS.output_dir,
          label_name,
          '%s-%02d-of-%02d.tfrecord' % (dataset, shard_id+1, _NUM_SHARDS))
      writer = tf.python_io.TFRecordWriter(output_filename)
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id+1)*num_per_shard, class_counts)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i+1, class_counts, shard_id+1))
        sys.stdout.flush()

        image_id = image_ids[i]
        label = list(ori_data[ori_data['Id']==image_id]['Target'])[0]
        if label_idx not in label:
          raise RuntimeError('Label is wrong.')
        one_hot_label = np.zeros([_NUM_CLASSES], np.int64)
        for l in label:
          one_hot_label[l] = 1
        image_filename = os.path.join(FLAGS.dataset_folder, dataset, image_id)
        image_data_green = tf.gfile.FastGFile(image_filename+'_green'+'.png', 'rb').read()
        image_data_red = tf.gfile.FastGFile(image_filename+'_red'+'.png', 'rb').read()
        image_data_blue = tf.gfile.FastGFile(image_filename+'_blue'+'.png', 'rb').read()
        image_data_yellow = tf.gfile.FastGFile(image_filename+'_yellow'+'.png', 'rb').read()
        example = build_dataset.protein_to_tfexample(image_data_green, image_data_red, 
                                                     image_data_blue, image_data_yellow, 
                                                     one_hot_label, image_id, image_format='png')
        writer.write(example.SerializeToString())
      sys.stdout.write('\n--- Split size: %d ---\n'%(end_idx-start_idx))
      sys.stdout.flush()
  sys.stdout.write('[*] Number of total samples: %d\n' % (num_samples))
  sys.stdout.flush()


def main():
  train_data = pd.read_csv(os.path.join(FLAGS.dataset_folder, 'train.csv'))
  for key in _IDX_TO_NAME.keys():
    train_data[_IDX_TO_NAME[key]] = 0
  train_data = train_data.apply(fill_targets, axis=1)
  
  per_class_counts = {}
  per_class_image_ids = {}
  sub_data = train_data.drop(['Target'],axis=1).copy(deep=True)
  target_counts = sub_data.drop(['Id'],axis=1).sum(axis=0).sort_values(ascending=False)
  for i in range(_NUM_CLASSES):
    special_target = target_counts.keys()[-1]
    per_class_counts[special_target] = target_counts[-1]
    per_class_image_ids[special_target] = list(sub_data['Id'][sub_data[special_target] == 1])
    sub_data = sub_data[sub_data[special_target] == 0].drop([special_target],axis=1)
    target_counts = sub_data.drop(['Id'],axis=1).sum(axis=0).sort_values(ascending=False)

  convert_tfrecord('protein', train_data, per_class_image_ids, per_class_counts)

if __name__ == '__main__':
  main()