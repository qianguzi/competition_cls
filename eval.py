from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import os, math
import tensorflow as tf
from tensorflow.contrib import slim

import common, model
from net.mobilenet import mobilenet_v2
#from dataset.get_lcz_dataset import get_dataset
from dataset import get_dataset

flags = tf.app.flags

flags.DEFINE_string('master', '', 'Session master')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_integer('image_size', 96, 'Input image resolution')
flags.DEFINE_string('checkpoint_dir', './train_log', 'The directory for checkpoints')
flags.DEFINE_string('eval_dir', './val_log', 'Directory for writing eval event logs')
flags.DEFINE_string('dataset_dir', '/mnt/home/hdd/hdd1/home/junq/dataset', 'Location of dataset.')
#flags.DEFINE_string('dataset_dir', '/media/deeplearning/f3cff4c9-1ab9-47f0-8b82-231dedcbd61b/lcz/tfrecord/',
#                    'Location of dataset.')
flags.DEFINE_string('dataset', 'name', 'Name of the dataset.')
flags.DEFINE_string('eval_split', 'val',
                    'Which split of the dataset used for evaluation')
flags.DEFINE_integer('eval_interval_secs', 60 * 6,
                     'How often (in seconds) to run evaluation.')
flags.DEFINE_integer('max_number_of_evaluations', 500,
                     'Maximum number of eval iterations. Will loop '
                     'indefinitely upon nonpositive values.')
flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

FLAGS = flags.FLAGS

def metrics(end_points, labels):
  """Specify the metrics for eval.
  Args:
    end_points: Include predictions output from the graph.
    labels: Ground truth labels for inputs.
  Returns:
     Eval Op for the graph.
  """
  # Define the evaluation metric.
  metric_map = {}
  if FLAGS.multi_label:
    predictions = tf.where(end_points['Predictions']>0.5, 1, 0)
  else:
    predictions = tf.argmax(end_points['Predictions'], axis=1)
    predictions = tf.reshape(predictions, shape=[-1])
    labels_id = tf.argmax(labels, axis=1)
    labels_id = tf.reshape(labels_id, shape=[-1])
    metric_map['accuracy'] = tf.metrics.accuracy(labels_id, predictions)
    predictions = slim.one_hot_encoding(predictions, FLAGS.num_classes)
  for i in range(FLAGS.num_classes):
    metric_map['accuracy_%02d'%i]=tf.metrics.accuracy(labels[:,i], predictions[:,i])
    metric_map['accuracy_%02d'%i]=tf.metrics.precision(labels[:,i], predictions[:,i])
    metric_map['accuracy_%02d'%i]=tf.metrics.recall(labels[:,i], predictions[:,i])

  metrics_to_values, metrics_to_updates = (
      tf.contrib.metrics.aggregate_metric_map(metric_map))

  for metric_name, metric_value in six.iteritems(metrics_to_values):
    slim.summaries.add_scalar_summary(metric_value, metric_name, prefix='accuracy', print_summary=True)
  return list(metrics_to_updates.values())


def eval_model():
  """Evaluates model."""
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  tf.logging.info('Evaluating on %s set', FLAGS.eval_split)
  g = tf.Graph()
  with g.as_default():
    samples, num_samples = get_dataset.get_dataset(FLAGS.dataset, FLAGS.dataset_dir,
                                         split_name=FLAGS.val_split,
                                         is_training=False,
                                         image_size=[FLAGS.image_size, FLAGS.image_size],
                                         batch_size=FLAGS.batch_size,
                                         channel=FLAGS.input_channel)
    inputs = tf.identity(samples['data'], name='data')
    labels = tf.identity(samples['label'], name='label')
    model_options = common.ModelOptions(output_stride=FLAGS.output_stride)
    net, end_points = model.get_features(
        inputs,
        model_options=model_options,
        is_training=False,
        fine_tune_batch_norm=False)

    _, end_points = model.classification(net, end_points, 
                                         num_classes=FLAGS.num_classes,
                                         is_training=False)
    eval_ops = metrics(end_points, labels)
    #num_samples = 1000
    num_batches = math.ceil(num_samples / float(FLAGS.batch_size))
    tf.logging.info('Eval num images %d', num_samples)
    tf.logging.info('Eval batch size %d and num batch %d',
                    FLAGS.batch_size, num_batches)
    num_eval_iters = None
    if FLAGS.max_number_of_evaluations > 0:
      num_eval_iters = FLAGS.max_number_of_evaluations
    session_config = tf.ConfigProto(device_count={'GPU': 0})
    slim.evaluation.evaluation_loop(
        FLAGS.master,
        FLAGS.checkpoint_dir,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=eval_ops,
        session_config=session_config,
        max_number_of_evaluations=num_eval_iters,
        eval_interval_secs=FLAGS.eval_interval_secs)


def main(unused_arg):
  eval_model()


if __name__ == '__main__':
  tf.app.run(main)