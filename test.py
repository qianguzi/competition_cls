import os, h5py
import numpy as np
import tensorflow as tf
from time import time

from dataset import common

flags = tf.app.flags

flags.DEFINE_string('test_dataset_path',
                    '/home/data/lcz/test/round1_test_a_20181109.h5',
                    'Folder containing dataset.')
flags.DEFINE_integer('batch_size', 100, 'Batch size')
flags.DEFINE_string('save_path', './result/submission.csv',
                    'Path to output submission file.')
FLAGS = flags.FLAGS

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
    num_test = s1_test.shape[0]
    num_batch = int(np.ceil(num_test/FLAGS.batch_size))

    with tf.Session() as sess:
        sess.run(init_op)
        pred_rows = []
        start_time = time()
        for idx in range(num_batch):
          single_start_time = time()
          try:
            s1_data = s1_test[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
            s2_data = s2_test[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
          except:
            s1_data = s1_test[idx*FLAGS.batch_size:]
            s2_data = s2_test[idx*FLAGS.batch_size:]
          img_data = common.preprocess_fn(s1_data, s2_data)

          pred = sess.run(prediction, {img_tensor: img_data})
          
          pred = pred[:, 1:].astype(np.uint8)
          pred_rows.append(pred)
          print('Batch[{0}/{1}] time cost: {2}'.format(idx+1, num_batch, time()-single_start_time))
        pred_rows = np.concatenate(pred_rows, 0)
        print('All time cost: %s' % (time()-start_time))
        tf.gfile.MakeDirs(os.path.dirname(FLAGS.save_path))
        np.savetxt(FLAGS.save_path, pred_rows, delimiter=",", fmt='%s')
        print('File submission.csv success saved.')


if __name__ == '__main__':
  model_test()