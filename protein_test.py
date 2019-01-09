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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

_THRESHOULD = [0.00000001, 0.0968, 0.2178, 0.0131, 0.0055, 0.1866, 0.2934,
               0.4926, 0.1592, 0.0120, 0.3293, 0.2142, 0.2788, 0.4694,
               0.0053, 0.1375, 0.2838, 0.4881, 0.2479, 0.0670, 0.2771,
               0.0092, 0.3076, 0.0194, 0.2156, 0.0004, 0.3336, 0.3301]

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
          if len(predictions_id) == 0:
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
