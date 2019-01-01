from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.tools import freeze_graph

from net.mobilenet import mobilenet_v2
from dataset import data_augmentation
import common, model

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_path', './train_log/model.ckpt-193793', 'Checkpoint path')
flags.DEFINE_string('export_path', './result/protein/model.pb',
                    'Path to output Tensorflow frozen graph.')
flags.DEFINE_multi_integer('input_shape', [512, 512, 4], 'The shape of input image.')
flags.DEFINE_integer('channel', 0, 'Number of channel.')
flags.DEFINE_integer('image_size', 256, 'Input image resolution')
flags.DEFINE_integer('output_stride', 32,
                     'The ratio of input to output spatial resolution.')
# Input name of the exported model.
_INPUT_NAME = 'ImageTensor'

# Output name of the exported model.
_OUTPUT_NAME = 'Prediction'
_OUTPUT_COUNTS_NAME = 'CountsPrediction'


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info('Prepare to export model to: %s', FLAGS.export_path)

  with tf.Graph().as_default():
    input_image = tf.placeholder(tf.float32, FLAGS.input_shape, name=_INPUT_NAME)
    inputs = data_augmentation.preprocess_image(
        input_image, FLAGS.image_size, FLAGS.image_size, is_training=False)
    if FLAGS.channel:
      #inputs = inputs[:,:,:FLAGS.channel]
      inputs = inputs[:,:,3:]
    inputs = tf.expand_dims(inputs, 0)
    model_options = common.ModelOptions(output_stride=FLAGS.output_stride)
    net, end_points = model.get_features(
        inputs,
        model_options=model_options,
        is_training=False,
        fine_tune_batch_norm=False)

    if FLAGS.hierarchical_cls:
      end_points = model.hierarchical_classification(net, end_points, is_training=False)
    else:
      _, end_points = model.classification(net, end_points, 
                                           num_classes=FLAGS.num_classes,
                                           is_training=False)
    if FLAGS.add_counts_logits:
      _, end_points = model.classification(net, end_points, num_classes=6,
                                              is_training=False, scope='Counts_logits')
    #prediction = tf.argmax(end_points['Predictions'], 1)
    #prediction = slim.one_hot_encoding(prediction, FLAGS.num_classes)
    prediction = tf.identity(end_points['Logits_Predictions'], name=_OUTPUT_NAME)
    counts_prediction = tf.identity(end_points['Counts_logits_Predictions'], name=_OUTPUT_COUNTS_NAME)

    saver = tf.train.Saver(tf.model_variables())

    tf.gfile.MakeDirs(os.path.dirname(FLAGS.export_path))
    freeze_graph.freeze_graph_with_def_protos(
        tf.get_default_graph().as_graph_def(add_shapes=True),
        saver.as_saver_def(),
        FLAGS.checkpoint_path,
        _OUTPUT_NAME+','+_OUTPUT_COUNTS_NAME,
        restore_op_name=None,
        filename_tensor_name=None,
        output_graph=FLAGS.export_path,
        clear_devices=True,
        initializer_nodes=None)


if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_path')
  flags.mark_flag_as_required('export_path')
  tf.app.run()
