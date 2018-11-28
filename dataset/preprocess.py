import os
import h5py, glob
import numpy as np
import tensorflow as tf


def data_norm(s, s_mean, s_std):
  s_norm = []
  for i in range(18):
    if len(s.shape) == 3:
      s_c = (s[:,:,i]-s_mean[i])/s_std[i]
    elif len(s.shape) == 4:
      s_c = (s[:,:,:,i]-s_mean[i])/s_std[i]
    else:
      raise ValueError('The dimensions of s must be 3/4.')
    if (i==4 or i==5):
      s_c = np.where(s_c>4, 4, s_c)
      s_c = np.where(s_c<-4, -4, s_c)/4
    elif (i==6 or i==7):
      s_c = np.where(s_c>0.1, 0.1, s_c)
      s_c = np.where(s_c<-0.1, -0.1, s_c)*10
    else:
      s_c = np.where(s_c>3, 3, s_c)
      s_c = np.where(s_c<-3, -3, s_c)/3
    s_norm.append(s_c)
  s_norm = np.stack(s_norm, -1)
  return s_norm

def data_zoom(s1, s2):
  s1 = np.where(s1>10, 10, s1)
  s1 = np.where(s1<-10, -10, s1)

  s2 = np.where(s2>2.5, 2.5, s2)
  s2 = np.where(s2>0.4, 0.4+(s2-0.4)/10, s2)
  s = np.concatenate([s1,s2], -1)
  return s


def h5_read(path):
  fid = h5py.File(path,'r')
  s1 = fid['sen1']
  s2 = fid['sen2']
  label = fid['label']
  return s1, s2, label

def read_tfrecord(file_path, shuffle=True):
  reader = tf.TFRecordReader()

  file_path = glob.glob(file_path)
  filename_queue = tf.train.string_input_producer(
      file_path, shuffle=shuffle)

  _, serialized_example = reader.read(filename_queue)

  features = tf.parse_single_example(
      serialized_example,
      features={'data': tf.FixedLenFeature([32, 32, 18], tf.float32),
                'label': tf.FixedLenFeature([], tf.int64),
                'idx': tf.FixedLenFeature([], tf.int64)})
  sample = {
      'data': features['data'],
      'label': features['label'],
      'idx': features['idx']
  }

  return sample

def get_batch(file_path, batch_size, is_training=True):
  '''Get batch.'''
  sample = read_tfrecord(file_path, shuffle=is_training)
  capacity = 5 * batch_size

  samples = tf.train.batch(sample,
                           batch_size, 
                           capacity=capacity,
                           num_threads=4,
                           allow_smaller_final_batch=not is_training)
  return samples

def _main(base_dir):
  path_training = os.path.join(base_dir, 'training.h5')
  s1_training, s2_training, label_training = h5_read(path_training)

  s1 = []
  s2 = []
  for i in range(17):
    idx = np.where(label_training[:,i])[0][:500]
    s1.append(s1_training[list(idx)])
    s2.append(s2_training[list(idx)])
  s1 = np.concatenate(s1)
  s2 = np.concatenate(s2)
  s = data_zoom(s1, s2)
  s = s.reshape([-1, 18])
  s[:,4:6] = np.log10(s[:,4:6])
  s_mean = np.mean(s, 0)
  s_std = np.std(s, 0)
  print(s_mean, s_std)

if __name__ == '__main__':
  base_dir = '/home/data/lcz'
  _main(base_dir)