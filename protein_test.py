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

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

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

FLAGS = flags.FLAGS

_THRESHOULD = [0.0457, 0.0953, 0.0921, 0.1265, 0.0149, 0.3209, 0.1713, 
               0.1061, 0.6000, 0.1421, 0.6000, 0.6000, 0.6000, 0.6000,
               0.1665, 0.6000, 0.3017, 0.6000, 0.3249, 0.2305, 0.0717,
               0.1269, 0.3593, 0.0737, 0.1681, 0.0653, 0.6000, 0.6000]

def model_test():
  g = tf.Graph()
  with g.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(os.path.join(FLAGS.save_path, 'model.pb'), 'rb') as f:
        od_graph_def.ParseFromString(f.read())
        img_tensor, prediction = tf.import_graph_def(
                od_graph_def,
                return_elements=['ImageTensor:0', 'Prediction:0'])
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

          logits_np = sess.run(prediction, {img_tensor: image_data})

          logits_np = logits_np[0]
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
