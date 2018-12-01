import os, math
import tensorflow as tf
from tensorflow.contrib import slim

import common
from net.mobilenet import mobilenet_v2
from dataset.get_dataset import get_dataset

flags = tf.app.flags

flags.DEFINE_string('master', '', 'Session master')
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('image_size', 64, 'Input image resolution')
flags.DEFINE_bool('quantize', False, 'Quantize training')
flags.DEFINE_string('checkpoint_dir', './train_log', 'The directory for checkpoints')
flags.DEFINE_string('eval_dir', './val_log', 'Directory for writing eval event logs')
flags.DEFINE_string('dataset_dir', '/media/jun/data/lcz/tfrecord', 'Location of dataset.')
#flags.DEFINE_string('dataset_dir', '/media/deeplearning/f3cff4c9-1ab9-47f0-8b82-231dedcbd61b/lcz/tfrecord/',
#                    'Location of dataset.')
flags.DEFINE_string('dataset', 'default', 'Name of the dataset.')
flags.DEFINE_string('eval_split', 'val',
                    'Which split of the dataset used for evaluation')
flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                     'How often (in seconds) to run evaluation.')
flags.DEFINE_integer('max_number_of_evaluations', 5,
                     'Maximum number of eval iterations. Will loop '
                     'indefinitely upon nonpositive values.')

FLAGS = flags.FLAGS

def metrics(logits, labels):
  """Specify the metrics for eval.
  Args:
    logits: Logits output from the graph.
    labels: Ground truth labels for inputs.
  Returns:
     Eval Op for the graph.
  """
  labels = tf.squeeze(labels)
  names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
      'Accuracy': tf.metrics.accuracy(tf.argmax(logits, 1), labels),
      'Recall_5': tf.metrics.recall_at_k(labels, logits, 5),
  })
  for name, value in names_to_values.items():
    slim.summaries.add_scalar_summary(
        value, name, prefix='eval', print_summary=True)
  return list(names_to_updates.values())


def eval_model():
  """Evaluates model."""
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  tf.logging.info('Evaluating on %s set', FLAGS.eval_split)
  g = tf.Graph()
  with g.as_default():
    samples, num_samples = get_dataset(
      FLAGS.dataset, FLAGS.eval_split, FLAGS.dataset_dir, FLAGS.batch_size, is_training=False)
    inputs = tf.image.resize_images(samples['data'], [FLAGS.image_size, FLAGS.image_size])
    inputs = tf.identity(inputs, name='data')
    labels = tf.identity(samples['label'], name='label')
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
      _, end_points = mobilenet_v2.mobilenet(
          inputs,
          is_training=False,
          depth_multiplier=FLAGS.depth_multiplier,
          num_classes=FLAGS.num_classes,
          finegrain_classification_mode=True)

    if FLAGS.quantize:
      tf.contrib.quantize.create_eval_graph()

    eval_ops = metrics(end_points['Predictions'], labels)

    num_batches = math.ceil(num_samples / float(FLAGS.batch_size))
    tf.logging.info('Eval num images %d', num_samples)
    tf.logging.info('Eval batch size %d and num batch %d',
                    FLAGS.batch_size, num_batches)
    num_eval_iters = None
    if FLAGS.max_number_of_evaluations > 0:
      num_eval_iters = FLAGS.max_number_of_evaluations
    slim.evaluation.evaluation_loop(
        FLAGS.master,
        FLAGS.checkpoint_dir,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=eval_ops,
        max_number_of_evaluations=num_eval_iters,
        eval_interval_secs=FLAGS.eval_interval_secs)


def main(unused_arg):
  eval_model()


if __name__ == '__main__':
  tf.app.run(main)