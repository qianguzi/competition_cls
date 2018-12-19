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
from utils import f1_score

flags = tf.app.flags

flags.DEFINE_string('master', '', 'Session master')
flags.DEFINE_integer('batch_size', 2, 'Batch size')
flags.DEFINE_integer('image_size', 112, 'Input image resolution')
flags.DEFINE_string('checkpoint_dir', './train_log', 'The directory for checkpoints')
flags.DEFINE_string('eval_dir', '/mnt/home/hdd/hdd1/home/junq/lcz/val_log', 'Directory for writing eval event logs')
flags.DEFINE_string('dataset_dir', '/mnt/home/hdd/hdd1/home/junq/dataset', 'Location of dataset.')
#flags.DEFINE_string('dataset_dir', '/media/deeplearning/f3cff4c9-1ab9-47f0-8b82-231dedcbd61b/lcz/tfrecord/',
#                    'Location of dataset.')
flags.DEFINE_string('dataset', 'protein', 'Name of the dataset.')
flags.DEFINE_string('eval_split', 'protein-02',
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
  predictions = end_points['Predictions']
  if FLAGS.multi_label:
    predictions = tf.where(tf.greater_equal(predictions, 0.5),
                           tf.ones_like(predictions),
                           tf.zeros_like(predictions))
  else:
    predictions = tf.argmax(predictions, axis=1)
    predictions = tf.reshape(predictions, shape=[-1])
    labels_id = tf.argmax(labels, axis=1)
    labels_id = tf.reshape(labels_id, shape=[-1])
    metric_map['accuracy'] = tf.metrics.accuracy(labels_id, predictions)
    predictions = slim.one_hot_encoding(predictions, FLAGS.num_classes)
  subacc_list = []
  subf1_list = []
  for i in range(FLAGS.num_classes):
    metric_map['subacc/accuracy_%02d'%i] = tf.metrics.accuracy(labels[:,i], predictions[:,i])
    subacc_list.append(metric_map['subacc/accuracy_%02d'%i][0])
    metric_map['subf1/f1_score_%02d'%i] = tf.contrib.metrics.f1_score(labels[:,i], 
                                                                      end_points['Predictions'][:,i])
    subf1_list.append(metric_map['subf1/f1_score_%02d'%i][0])
  metric_map['ave_accuracy'] = tf.metrics.mean(tf.stack(subacc_list, 0))
  metric_map['f1_score'] = tf.metrics.mean(tf.stack(subf1_list, 0))
  counts_f1, update_f1 = f1_score.streaming_counts(labels, predictions, FLAGS.num_classes)
  micro_f1, macro_f1, weight_f1 = f1_score.streaming_f1(counts_f1)

  metrics_to_values, metrics_to_updates = (
      tf.contrib.metrics.aggregate_metric_map(metric_map))

  for metric_name, metric_value in six.iteritems(metrics_to_values):
    slim.summaries.add_scalar_summary(metric_value, metric_name, prefix='eval', print_summary=True)
  slim.summaries.add_scalar_summary(micro_f1, 'micro_f1', prefix='eval', print_summary=True)
  slim.summaries.add_scalar_summary(macro_f1, 'macro_f1', prefix='eval', print_summary=True)
  slim.summaries.add_scalar_summary(weight_f1, 'weight_f1', prefix='eval', print_summary=True)
  return list(metrics_to_updates.values()).append(update_f1)


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
    inputs = tf.identity(samples['image'], name='image')
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
    # session_config = tf.ConfigProto(device_count={'GPU': 0})
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.3
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