# pylint: disable=E1129, E1101
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os, cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from time import time

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

flags = tf.app.flags

flags.DEFINE_string('test_dataset_path',
                    # '/media/jun/data/protein',
                    '/mnt/home/hdd/hdd1/home/LiaoL/Kaggle/Protein/dataset',
                    'Folder containing dataset.')
# flags.DEFINE_string('test_dataset_path',
#                     './round1_test_a_20181109.h5',
#                     'Folder containing dataset.')
flags.DEFINE_string('save_path', './result/protein',
                    'Path to output submission file.')
# flags.DEFINE_float('threshould', 0.19, 'The momentum value to use')

FLAGS = flags.FLAGS

_THRESHOULD = [0.1853, 0.1357, 0.1221, 0.2865, 0.2245, 0.1601, 0.1177, 
               0.1733, 0.0081, 0.0005, 0.0005, 0.1473, 0.2457, 0.2321,
               0.2389, 0.0013, 0.0429, 0.1421, 0.1245, 0.2277, 0.1625,
               0.2141, 0.1061, 0.1745, 0.0417, 0.1541, 0.1345, 0.0029]

def model_test():
  g = tf.Graph()
  with g.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(os.path.join(FLAGS.save_path, 'model.pb'), 'rb') as f:
        od_graph_def.ParseFromString(f.read())
        img_tensor, prediction, counts_prediction = tf.import_graph_def(
                od_graph_def,
                return_elements=['ImageTensor:0', 'Prediction:0', 'CountsPrediction:0'])
    init_op = tf.global_variables_initializer()
    test_data = pd.read_csv(os.path.join(FLAGS.test_dataset_path, 'sample_submission.csv'))
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    with tf.Session(config=session_config) as sess:
        sess.run(init_op)
        pred_rows = []
        start_time = time()
        for i, image_id in enumerate(test_data['Id']):
          image_filename = os.path.join(FLAGS.test_dataset_path, 'test', image_id)
          image_data_green = cv2.imread(image_filename+'_green'+'.png', 0)
          image_data_red = cv2.imread(image_filename+'_red'+'.png', 0)
          image_data_blue = cv2.imread(image_filename+'_blue'+'.png', 0)
          image_data_yellow = cv2.imread(image_filename+'_yellow'+'.png', 0)
          image_data = np.stack(
              [image_data_green, image_data_red, image_data_blue, image_data_yellow], axis=-1)
          image_data = (image_data / 255).astype(np.float32)

          logits_np, counts_np = sess.run([prediction, counts_prediction], {img_tensor: image_data})
          
          logits_np = logits_np[0]
          counts_np = counts_np[0]
          predictions_id = list(np.where(logits_np > _THRESHOULD)[0])
          if predictions_id is None:
            max_id = np.argmax(logits_np)
            predictions_id.append(max_id)
          predictions_id = ' '.join(str(x) for x in predictions_id)
          pred_rows.append({'Id': image_id, 'Predicted': predictions_id})
          sys.stdout.write('\r>> Data[{0}/{1}] time cost: {2}'.format(i+1, 11702, time()-start_time))
          sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        submission_df = pd.DataFrame(pred_rows)[['Id', 'Predicted']]
        submission_df.to_csv(os.path.join(FLAGS.save_path, 'submission.csv'), index=False)
        sys.stdout.write('[*]File submission.csv success saved.\n')
        sys.stdout.flush()


if __name__ == '__main__':
  model_test()
