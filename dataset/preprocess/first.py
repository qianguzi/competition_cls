from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import numpy as np

_DATASET_MEAN = [6.79034058e-05, 3.13958198e-05, 8.54824102e-05, -7.53851136e-05,
                -1.61852048e+00, -1.02397158e+00, -3.99997315e-04, 1.44433013e-03,
                 1.29802515e-01, 1.17745393e-01, 1.14519432e-01, 1.27774508e-01,
                 1.70839563e-01, 1.92380020e-01, 1.85248741e-01, 2.06286210e-01,
                 1.74647946e-01, 1.28752703e-01]
_DATASET_STD = [0.20595731, 0.20523204, 0.48476867, 0.48547576, 0.49312326, 0.54128573,
                0.25078281, 0.18858326, 0.0413942, 0.05197171, 0.07318039, 0.06928833,
                0.07388366, 0.08290952, 0.08446056, 0.09017361, 0.09451134, 0.09073909]


def h5_read(path):
  fid = h5py.File(path,'r')
  s1 = fid['sen1']
  s2 = fid['sen2']
  label = fid['label']
  return s1, s2, label

def first_preprocess(s1_data, s2_data):
  img_data = data_zoom(s1_data, s2_data)
  img_data[:,:,4:6] = np.log10(img_data[:,:,4:6])
  img_data = data_norm(img_data, _DATASET_MEAN, _DATASET_STD)
  return img_data


def data_norm(s, s_mean, s_std):
  s_norm = []
  for i in range(18):
    if len(s.shape) == 3:
      s_c = (s[:,:,i]-s_mean[i])/s_std[i]
    elif len(s.shape) == 4:
      s_c = (s[:,:,:,i]-s_mean[i])/s_std[i]
    else:
      raise ValueError('The dimensions of s must be 3 or 4.')
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