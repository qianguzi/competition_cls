from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import os, math
import tensorflow as tf
from tensorflow.contrib import slim

import common, model
from net.mobilenet import mobilenet_v2
from dataset.get_dataset import get_dataset

flags = tf.app.flags

flags.DEFINE_string('master', '', 'Session master')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_integer('image_size', 96, 'Input image resolution')
flags.DEFINE_string('checkpoint_dir', './train_log', 'The directory for checkpoints')
flags.DEFINE_string('eval_dir', './val_log', 'Directory for writing eval event logs')
flags.DEFINE_string('dataset_dir', '/media/jun/data/lcz/tfrecord', 'Location of dataset.')
#flags.DEFINE_string('dataset_dir', '/media/deeplearning/f3cff4c9-1ab9-47f0-8b82-231dedcbd61b/lcz/tfrecord/',
#                    'Location of dataset.')
flags.DEFINE_string('dataset', 'name', 'Name of the dataset.')
flags.DEFINE_string('eval_split', 'val',
                    'Which split of the dataset used for evaluation')
flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                     'How often (in seconds) to run evaluation.')
flags.DEFINE_integer('max_number_of_evaluations', 500,
                     'Maximum number of eval iterations. Will loop '
                     'indefinitely upon nonpositive values.')
flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

FLAGS = flags.FLAGS

def metrics(logits, labels):
  """Specify the metrics for eval.
  Args:
    logits: Logits output from the graph.
    labels: Ground truth labels for inputs.
  Returns:
     Eval Op for the graph.
  """
  predictions = tf.argmax(logits, axis=1)
  predictions = tf.reshape(predictions, shape=[-1])
  labels = tf.reshape(labels, shape=[-1])

  # Define the evaluation metric.
  metric_map = {}
  metric_map['accuracy'] = tf.metrics.accuracy(labels, predictions)

  metrics_to_values, metrics_to_updates = (
      tf.contrib.metrics.aggregate_metric_map(metric_map))

  for metric_name, metric_value in six.iteritems(metrics_to_values):
    slim.summaries.add_scalar_summary(metric_value, metric_name, print_summary=True)
  return list(metrics_to_updates.values())


def eval_model():
  """Evaluates model."""
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  tf.logging.info('Evaluating on %s set', FLAGS.eval_split)
  g = tf.Graph()
  with g.as_default():
    samples, num_samples = get_dataset(FLAGS.dataset, FLAGS.eval_split, FLAGS.dataset_dir,
                                       FLAGS.image_size, FLAGS.batch_size, is_training=False)
    inputs = tf.identity(samples['data'], name='data')
    labels = tf.identity(samples['label'], name='label')
    model_options = common.ModelOptions(output_stride=FLAGS.output_stride)
    _, end_points = model.get_logits(
        inputs,
        model_options=model_options,
        num_classes=FLAGS.num_classes,
        is_training=False,
        fine_tune_batch_norm=False)

    eval_ops = metrics(end_points['Predictions'], labels)
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