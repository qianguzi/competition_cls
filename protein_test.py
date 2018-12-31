from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os, cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from time import time

flags = tf.app.flags

flags.DEFINE_string('test_dataset_path',
                    # '/media/jun/data/protein',
                    '/mnt/home/hdd/hdd1/home/LiaoL/Kaggle/Protein/dataset',
                    'Folder containing dataset.')
#flags.DEFINE_string('test_dataset_path',
#                    './round1_test_a_20181109.h5',
#                    'Folder containing dataset.')
flags.DEFINE_string('save_path', './result/protein',
                    'Path to output submission file.')
flags.DEFINE_float('threshould', 0.19, 'The momentum value to use')

FLAGS = flags.FLAGS

def model_test():
  g = tf.Graph()
  with g.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(os.path.join(FLAGS.save_path, 'model-25653.pb'), 'rb') as f:
        od_graph_def.ParseFromString(f.read())
        img_tensor, prediction, counts_prediction = tf.import_graph_def(
                od_graph_def,
                return_elements=['ImageTensor:0', 'Prediction:0', 'CountsPrediction:0'])
    init_op = tf.global_variables_initializer()
    test_data = pd.read_csv(os.path.join(FLAGS.test_dataset_path, 'sample_submission.csv'))
    with tf.Session() as sess:
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
          
          logits_np = logits_np[0][1:]
          counts_np = counts_np[0]
          sorted_ids = np.argsort(logits_np)
          counts = np.argmax(counts_np)
          prediction_id = sorted_ids[(-counts):]
          prediction_id = ' '.join(str(x) for x in prediction_id)
          pred_rows.append({'Id': image_id, 'Predicted': prediction_id})
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