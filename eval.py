import math
import tensorflow as tf
from tensorflow.contrib import slim

from mobilenet import mobilenet_v2
from dataset import preprocess

flags = tf.app.flags

flags.DEFINE_string('master', '', 'Session master')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
flags.DEFINE_integer('num_classes', 18, 'Number of classes to distinguish')
flags.DEFINE_integer('num_examples', 24119, 'Number of examples to evaluate')
flags.DEFINE_integer('image_size', 32, 'Input image resolution')
flags.DEFINE_float('depth_multiplier', 1.5, 'Depth multiplier for mobilenet')
flags.DEFINE_bool('quantize', False, 'Quantize training')
flags.DEFINE_string('checkpoint_dir', './train_log', 'The directory for checkpoints')
flags.DEFINE_string('eval_dir', './val_log', 'Directory for writing eval event logs')
flags.DEFINE_string('dataset_dir', '/media/jun/data/lcz/tfrecord/val-*', 'Location of dataset')
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


def build_model():
  """Build the mobilenet_v1 model for evaluation.
  Returns:
    g: graph with rewrites after insertion of quantization ops and batch norm
    folding.
    eval_ops: eval ops for inference.
    variables_to_restore: List of variables to restore from checkpoint.
  """
  g = tf.Graph()
  with g.as_default():
    samples = preprocess.get_batch(FLAGS.dataset_dir, FLAGS.batch_size, is_training=False)
    inputs = tf.identity(samples['data'], name='data')
    labels = tf.identity(samples['label'], name='label')
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
      logits, _ = mobilenet_v2.mobilenet(
          inputs,
          is_training=False,
          depth_multiplier=FLAGS.depth_multiplier,
          num_classes=FLAGS.num_classes)

    if FLAGS.quantize:
      tf.contrib.quantize.create_eval_graph()

    eval_ops = metrics(logits, labels)

  return g, eval_ops


def eval_model():
  """Evaluates mobilenet_v1."""
  tf.logging.set_verbosity(tf.logging.INFO)
  g, eval_ops = build_model()
  with g.as_default():
    num_batches = math.ceil(FLAGS.num_examples / float(FLAGS.batch_size))
    tf.logging.info('Eval num images %d', FLAGS.num_examples)
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