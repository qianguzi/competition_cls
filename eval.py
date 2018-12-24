from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import os, math
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

import common, model
from net.mobilenet import mobilenet_v2
#from dataset.get_lcz_dataset import get_dataset
from dataset import get_dataset
from utils import streaming_f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags = tf.app.flags

flags.DEFINE_string('master', '', 'Session master')
flags.DEFINE_integer('batch_size', 1, 'Batch size')
flags.DEFINE_integer('image_size', 112, 'Input image resolution')
# flags.DEFINE_string('checkpoint_dir', '/mnt/home/hdd/hdd1/home/junq/lcz/train_log', 'The directory for checkpoints')
# flags.DEFINE_string('eval_dir', '/mnt/home/hdd/hdd1/home/junq/lcz/val_log', 'Directory for writing eval event logs')
# flags.DEFINE_string('dataset_dir', '/mnt/home/hdd/hdd1/home/junq/dataset', 'Location of dataset.')
flags.DEFINE_string('checkpoint_dir', '/home/jun/mynb/lcz/train_log/model.ckpt-98847', 'The directory for checkpoints')
flags.DEFINE_string('eval_dir', '/home/jun/mynb/lcz/val_log', 'Directory for writing eval event logs')
flags.DEFINE_string('dataset_dir', '/media/jun/data/tfrecord', 'Location of dataset.')
#flags.DEFINE_string('dataset_dir', '/media/deeplearning/f3cff4c9-1ab9-47f0-8b82-231dedcbd61b/lcz/tfrecord/',
#                    'Location of dataset.')
flags.DEFINE_string('dataset', 'protein', 'Name of the dataset.')
flags.DEFINE_string('eval_split', 'protein-002',
                    'Which split of the dataset used for evaluation')
flags.DEFINE_integer('eval_interval_secs', 60 * 6,
                     'How often (in seconds) to run evaluation.')
flags.DEFINE_integer('max_number_of_evaluations', 500,
                     'Maximum number of eval iterations. Will loop '
                     'indefinitely upon nonpositive values.')
flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')
flags.DEFINE_boolean('use_slim', False,
                     'Whether to use slim for eval or not.')
flags.DEFINE_float('threshould', 0.19, 'The momentum value to use')

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
  labels = tf.cast(labels, tf.int64)
  if FLAGS.multi_label:
    predictions = tf.where(tf.greater_equal(predictions, FLAGS.threshould),
                           tf.ones_like(predictions),
                           tf.zeros_like(predictions))
    predictions = tf.cast(predictions, tf.int64)
    counts_f1, update_f1 = streaming_f1_score.streaming_counts(labels, predictions, FLAGS.num_classes)
    micro_f1, macro_f1, weight_f1 = streaming_f1_score.streaming_f1(counts_f1)
    slim.summaries.add_scalar_summary(micro_f1, 'micro_f1', prefix='eval', print_summary=True)
    slim.summaries.add_scalar_summary(macro_f1, 'macro_f1', prefix='eval', print_summary=True)
    slim.summaries.add_scalar_summary(weight_f1, 'weight_f1', prefix='eval', print_summary=True)
    return update_f1
  else:
    predictions = tf.argmax(predictions, axis=1)
    predictions = tf.reshape(predictions, shape=[-1])
    labels_id = tf.argmax(labels, axis=1)
    labels_id = tf.reshape(labels_id, shape=[-1])
    metric_map['accuracy'] = tf.metrics.accuracy(labels_id, predictions)
    metrics_to_values, metrics_to_updates = (
        tf.contrib.metrics.aggregate_metric_map(metric_map))

    for metric_name, metric_value in six.iteritems(metrics_to_values):
      slim.summaries.add_scalar_summary(metric_value, metric_name, prefix='eval', print_summary=False)

    return list(metrics_to_updates.values())
  # subacc_list = []
  # subf1_list = []
  # for i in range(FLAGS.num_classes):
  #   metric_map['subacc/accuracy_%02d'%i] = tf.metrics.accuracy(labels[:,i], predictions[:,i])
  #   subacc_list.append(metric_map['subacc/accuracy_%02d'%i][0])
  #   metric_map['subpre/precision_%02d'%i] = tf.metrics.precision(labels[:,i], predictions[:,i])
  #   metric_map['subrec/recall_%02d'%i] = tf.metrics.recall(labels[:,i], predictions[:,i])
  #   class_pre = metric_map['subpre/precision_%02d'%i][0]
  #   class_rec = metric_map['subrec/recall_%02d'%i][0]
  #   class_f1 = (2 * class_pre * class_rec) / (class_pre + class_rec)
  #   subf1_list.append(class_f1)
  # ave_accuracy = tf.reduce_mean(tf.stack(subacc_list, 0))
  # f1_score = tf.reduce_mean(tf.stack(subf1_list, 0))
  # slim.summaries.add_scalar_summary(ave_accuracy, 'ave_accuracy', prefix='eval', print_summary=True)
  # slim.summaries.add_scalar_summary(f1_score, 'f1_score', prefix='eval', print_summary=True)


def get_checkpoint_init_fn(fine_tune_checkpoint, include_var=None, exclude_var=None):
    """Returns the checkpoint init_fn if the checkpoint is provided."""
    variables_to_restore = slim.get_variables_to_restore(include_var, exclude_var)
    slim_init_fn = slim.assign_from_checkpoint_fn(
        fine_tune_checkpoint,
        variables_to_restore,
        ignore_missing_vars=True)

    def init_fn(sess):
      slim_init_fn(sess)
    return init_fn


def eval_model():
  """Evaluates model."""
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  tf.logging.info('Evaluating on %s set', FLAGS.eval_split)
  g = tf.Graph()
  with g.as_default():
    samples, num_samples = get_dataset.get_dataset(FLAGS.dataset, FLAGS.dataset_dir,
                                                   split_name=FLAGS.eval_split,
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
    # session_config = tf.ConfigProto(device_count={'GPU': 0})
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.3
    if FLAGS.use_slim:
      num_eval_iters = None
      if FLAGS.max_number_of_evaluations > 0:
        num_eval_iters = FLAGS.max_number_of_evaluations
      slim.evaluation.evaluation_loop(
          FLAGS.master,
          FLAGS.checkpoint_dir,
          logdir=FLAGS.eval_dir,
          num_evals=num_batches,
          eval_op=eval_ops,
          session_config=session_config,
          max_number_of_evaluations=num_eval_iters,
          eval_interval_secs=FLAGS.eval_interval_secs)
    else:
      with tf.Session(config=session_config) as sess:
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        saver_fn = get_checkpoint_init_fn(FLAGS.checkpoint_dir)
        saver_fn(sess)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
          mean_iou_list = []
          i = 0
          while not coord.should_stop():
            logits_np, labels_np = sess.run([end_points['Predictions'], labels])
            logits_np = logits_np[0]
            labels_np = labels_np[0]
            results = np.argsort(logits_np)[-5:]
            result_logits = np.sort(logits_np)[-5:]
            labels_id = np.where(labels_np == 1)[0]
            i += 1
            print('Image[{0}]:\nlabels:{1}, \nresults: {2}, \nresult_logits: {3}'.format(i, labels_id, results, result_logits))
            max_id = np.argmax(logits_np)
            logits_np[max_id] = 1
            predictions_np = np.where(logits_np > FLAGS.threshould, 1, 0)
            insection = labels_np * predictions_np
            insection = np.sum(insection)
            union = np.where(labels_np + predictions_np > 0, 1, 0)
            union = np.sum(union)
            iou = insection / union
            mean_iou = np.mean(iou)
            mean_iou_list.append(mean_iou)
            all_mean_iou = np.mean(mean_iou_list)
            print('Image[{0}]--> iou: {1}, mean iou: {2}'.format(i, mean_iou, all_mean_iou))
        except tf.errors.OutOfRangeError:
          coord.request_stop()
          coord.join(threads)
        


def main(unused_arg):
  eval_model()


if __name__ == '__main__':
  tf.app.run(main)