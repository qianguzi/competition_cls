from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os, h5py
import numpy as np
import tensorflow as tf
from time import time

from dataset.preprocess import first, default

flags = tf.app.flags

flags.DEFINE_string('test_dataset_path',
                    '/home/data/lcz/test/round1_test_a_20181109.h5',
                    'Folder containing dataset.')
#flags.DEFINE_string('test_dataset_path',
#                    './round1_test_a_20181109.h5',
#                    'Folder containing dataset.')
flags.DEFINE_string('save_path', './fine_tune/submission-ensemble.csv',
                    'Path to output submission file.')
flags.DEFINE_string('preprocess_method', 'name', 'The image data preprocess term.')

FLAGS = flags.FLAGS


_PREPROCESS_METHOD = {
    'default': default.default_preprocess,
    'first': first.first_preprocess,
    'multiscale': default.new_preprocess,
    'name': default.new_preprocess,
}

def model_test():
  g = tf.Graph()
  with g.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.FastGFile('./result/model.pb', 'rb') as f:
        od_graph_def.ParseFromString(f.read())
        img_tensor, prediction= tf.import_graph_def(
                od_graph_def,
                return_elements=['ImageTensor:0', 'Prediction:0'])
    init_op = tf.global_variables_initializer()
    fid_test = h5py.File(FLAGS.test_dataset_path, 'r')
    s1_test = fid_test['sen1']
    s2_test = fid_test['sen2']
    num_test = int(s1_test.shape[0])
    
    preprocess_fn = _PREPROCESS_METHOD[FLAGS.preprocess_method]
    with tf.Session() as sess:
        sess.run(init_op)
        pred_rows = []
        start_time = time()
        for idx in range(num_test):
          s1_data = s1_test[idx]
          s2_data = s2_test[idx]
          img_data = preprocess_fn(s1_data, s2_data).astype(np.float32)

          pred = sess.run(prediction, {img_tensor: img_data})
          
          pred = pred[:, 1:].astype(np.uint8)
          pred_rows.append(pred)
          sys.stdout.write('\r>> Data[{0}/{1}] time cost: {2}'.format(idx+1, num_test, time()-start_time))
          sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        pred_rows = np.concatenate(pred_rows, 0)
        tf.gfile.MakeDirs(os.path.dirname(FLAGS.save_path))
        np.savetxt(FLAGS.save_path, pred_rows, delimiter=",", fmt='%s')
        sys.stdout.write('[*]File submission.csv success saved.\n')
        sys.stdout.flush()


def model_test_ensemble():
  g = tf.Graph()
  with g.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.FastGFile('./fine_tune/model-887849.pb', 'rb') as f:
        od_graph_def.ParseFromString(f.read())
        img_tensor_a, prediction_a= tf.import_graph_def(
                od_graph_def,
                return_elements=['ImageTensor:0', 'Prediction:0'])
    with tf.gfile.FastGFile('./fine_tune/model-803207.pb', 'rb') as f:
        od_graph_def.ParseFromString(f.read())
        img_tensor_b, prediction_b= tf.import_graph_def(
                od_graph_def,
                return_elements=['ImageTensor:0', 'Prediction:0'])
    with tf.gfile.FastGFile('./fine_tune/model-1583198.pb', 'rb') as f:
        od_graph_def.ParseFromString(f.read())
        img_tensor_c, prediction_c= tf.import_graph_def(
                od_graph_def,
                return_elements=['ImageTensor:0', 'Prediction:0'])
    with tf.gfile.FastGFile('./fine_tune/model-1850888.pb', 'rb') as f:
        od_graph_def.ParseFromString(f.read())
        img_tensor_d, prediction_d= tf.import_graph_def(
                od_graph_def,
                return_elements=['ImageTensor:0', 'Prediction:0'])
    init_op = tf.global_variables_initializer()
    fid_test = h5py.File(FLAGS.test_dataset_path, 'r')
    s1_test = fid_test['sen1']
    s2_test = fid_test['sen2']
    num_test = int(s1_test.shape[0])
    
    preprocess_fn = _PREPROCESS_METHOD[FLAGS.preprocess_method]
    with tf.Session() as sess:
        sess.run(init_op)
        pred_rows = []
        start_time = time()
        for idx in range(num_test):
          s1_data = s1_test[idx]
          s2_data = s2_test[idx]
          img_data = preprocess_fn(s1_data, s2_data).astype(np.float32)

          pred_a, pred_b, pred_c, pred_d = sess.run([prediction_a, prediction_b, prediction_c, prediction_d], 
                          {img_tensor_a: img_data, img_tensor_b: img_data, img_tensor_c: img_data, img_tensor_d: img_data})

          pred_logits = (pred_a[0, 1:] + pred_b[0, 1:] + pred_c[0, 1:] + pred_d[0, 1:]) / 4
          pred = np.zeros([17], np.uint8)
          pred[np.argmax(pred_logits)] = 1
          pred_rows.append(pred)
          sys.stdout.write('\r>> Data[{0}/{1}] time cost: {2}'.format(idx+1, num_test, time()-start_time))
          sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        #pred_rows = np.concatenate(pred_rows, 0)
        tf.gfile.MakeDirs(os.path.dirname(FLAGS.save_path))
        np.savetxt(FLAGS.save_path, pred_rows, delimiter=",", fmt='%s')
        sys.stdout.write('[*]File submission.csv success saved.\n')
        sys.stdout.flush()


if __name__ == '__main__':
  model_test_ensemble()