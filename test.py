import h5py, csv
import numpy as np
import pandas as pd
import tensorflow as tf
from time import time

from dataset import preprocess

_DATASET_MEAN = [6.79034058e-05, 3.13958198e-05, 8.54824102e-05, -7.53851136e-05,
                -1.61852048e+00, -1.02397158e+00, -3.99997315e-04, 1.44433013e-03,
                 1.29802515e-01, 1.17745393e-01, 1.14519432e-01, 1.27774508e-01,
                 1.70839563e-01, 1.92380020e-01, 1.85248741e-01, 2.06286210e-01,
                 1.74647946e-01, 1.28752703e-01]
_DATASET_STD = [0.20595731, 0.20523204, 0.48476867, 0.48547576, 0.49312326, 0.54128573,
                0.25078281, 0.18858326, 0.0413942, 0.05197171, 0.07318039, 0.06928833,
                0.07388366, 0.08290952, 0.08446056, 0.09017361, 0.09451134, 0.09073909]

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
    test_data_path = '/home/data/lcz/test/round1_test_a_20181109.h5'
    fid_test = h5py.File(test_data_path,'r')
    s1_test = fid_test['sen1']
    s2_test = fid_test['sen2']
    num_test = s1_test.shape[0]
    batch_size = 100
    num_batch = int(np.ceil(num_test/batch_size))

    with tf.Session() as sess:
        sess.run(init_op)
        pred_rows = []
        start_time = time()
        for idx in range(num_batch):
          single_start_time = time()
          try:
            s1_data = s1_test[idx*batch_size:(idx+1)*batch_size]
            s2_data = s2_test[idx*batch_size:(idx+1)*batch_size]
          except:
            s1_data = s1_test[idx*batch_size:]
            s2_data = s2_test[idx*batch_size:]
          img_data = preprocess.data_zoom(s1_data, s2_data)
          img_data[:,:,:,4:6] = np.log10(img_data[:,:,:,4:6])
          img_data = preprocess.data_norm(img_data, _DATASET_MEAN, _DATASET_STD)
          img_data = img_data.astype(np.float32)

          pred = sess.run(prediction, {img_tensor: img_data})
          
          pred = pred[:, 1:].astype(np.uint8)
          pred_rows.append(pred)
          print('Batch[{0}/{1}] time cost: {2}'.format(idx+1, num_batch, time()-single_start_time))
        pred_rows = np.concatenate(pred_rows, 0)
        print('All time cost: %s' % (time()-start_time))
        np.savetxt('./result/submission.csv', pred_rows, delimiter=",", fmt='%s')
        print('File submission.csv success saved.')


if __name__ == '__main__':
  model_test()