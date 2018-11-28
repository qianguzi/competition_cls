import os
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.tools import freeze_graph

from mobilenet import mobilenet_v2
from dataset import preprocess

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_path', './train_log/model.ckpt-7829', 'Checkpoint path')
flags.DEFINE_string('export_path', './result/model.pb',
                    'Path to output Tensorflow frozen graph.')
flags.DEFINE_float('depth_multiplier', 1.5, 'Depth multiplier for mobilenet')
flags.DEFINE_integer('num_classes', 18, 'Number of classes.')

# Input name of the exported model.
_INPUT_NAME = 'ImageTensor'

# Output name of the exported model.
_OUTPUT_NAME = 'Prediction'


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info('Prepare to export model to: %s', FLAGS.export_path)

  with tf.Graph().as_default():
    input_image = tf.placeholder(tf.float32, [None, 32, 32, 18], name=_INPUT_NAME)

    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
      _, end_points = mobilenet_v2.mobilenet(
          input_image,
          is_training=False,
          depth_multiplier=FLAGS.depth_multiplier,
          num_classes=FLAGS.num_classes)

    prediction = tf.argmax(end_points['Predictions'], 1)
    prediction = slim.one_hot_encoding(prediction, FLAGS.num_classes)
    prediction = tf.identity(prediction, name=_OUTPUT_NAME)

    saver = tf.train.Saver(tf.model_variables())

    tf.gfile.MakeDirs(os.path.dirname(FLAGS.export_path))
    freeze_graph.freeze_graph_with_def_protos(
        tf.get_default_graph().as_graph_def(add_shapes=True),
        saver.as_saver_def(),
        FLAGS.checkpoint_path,
        _OUTPUT_NAME,
        restore_op_name=None,
        filename_tensor_name=None,
        output_graph=FLAGS.export_path,
        clear_devices=True,
        initializer_nodes=None)


if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_path')
  flags.mark_flag_as_required('export_path')
  tf.app.run()
