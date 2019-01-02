from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six, sys
import os, math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import slim

import common, model
from net.mobilenet import mobilenet_v2
#from dataset.get_lcz_dataset import get_dataset
from dataset import get_dataset
from utils import streaming_f1_score
from sklearn.metrics import f1_score, precision_score, recall_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags = tf.app.flags

flags.DEFINE_string('master', '', 'Session master')
flags.DEFINE_integer('batch_size', 1, 'Batch size')
flags.DEFINE_integer('image_size', 224, 'Input image resolution')
# flags.DEFINE_string('checkpoint_dir', '/mnt/home/hdd/hdd1/home/junq/lcz/train_log', 'The directory for checkpoints')
# flags.DEFINE_string('eval_dir', '/mnt/home/hdd/hdd1/home/junq/lcz/val_log', 'Directory for writing eval event logs')
# flags.DEFINE_string('dataset_dir', '/mnt/home/hdd/hdd1/home/junq/dataset', 'Location of dataset.')
flags.DEFINE_string('checkpoint_dir', './result/model.ckpt-125715', 'The directory for checkpoints')
flags.DEFINE_string('eval_dir', './val_log', 'Directory for writing eval event logs')
# flags.DEFINE_string('dataset_dir', '/media/jun/data/tfrecord', 'Location of dataset.')
flags.DEFINE_string('dataset_dir', '../dataset', 'Location of dataset.')
#flags.DEFINE_string('dataset_dir', '/media/deeplearning/f3cff4c9-1ab9-47f0-8b82-231dedcbd61b/lcz/tfrecord/',
#                    'Location of dataset.')
flags.DEFINE_string('dataset', 'protein', 'Name of the dataset.')
flags.DEFINE_string('eval_split', 'protein',
                    'Which split of the dataset used for evaluation')
flags.DEFINE_integer('eval_interval_secs', 60 * 6,
                     'How often (in seconds) to run evaluation.')
flags.DEFINE_integer('max_number_of_evaluations', 500,
                     'Maximum number of eval iterations. Will loop '
                     'indefinitely upon nonpositive values.')
flags.DEFINE_integer('output_stride', 32,
                     'The ratio of input to output spatial resolution.')
flags.DEFINE_boolean('use_slim', False,
                     'Whether to use slim for eval or not.')
flags.DEFINE_integer('threshould', 2000, 'The momentum value to use')

FLAGS = flags.FLAGS

def metrics(end_points, labels, counts=None):
  """Specify the metrics for eval.
  Args:
    end_points: Include predictions output from the graph.
    labels: Ground truth labels for inputs.
    counts: Ground truth counts labels for inputs.
  Returns:
     Eval Op for the graph.
  """
  # Define the evaluation metric.
  metric_map = {}

  if FLAGS.add_counts_logits:
    predictions = end_points['Counts_logits_Predictions']
    labels = counts - 1
  else:
    predictions = end_points['Logits_Predictions']
    labels = tf.argmax(labels, axis=1)
  predictions = tf.argmax(predictions, axis=1)
  metric_map['accuracy'] = tf.metrics.accuracy(labels, predictions)
  metrics_to_values, metrics_to_updates = (
      tf.contrib.metrics.aggregate_metric_map(metric_map))

  for metric_name, metric_value in six.iteritems(metrics_to_values):
    slim.summaries.add_scalar_summary(metric_value, metric_name, prefix='eval', print_summary=True)

  return list(metrics_to_updates.values())


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
    if FLAGS.add_counts_logits:
      counts = tf.identity(samples['counts'], name='counts')
      _, end_points = model.classification(net, end_points, num_classes=5,
                                           is_training=False, scope='Counts_logits')
      eval_ops = metrics(end_points, labels, counts)
    else:
      eval_ops = metrics(end_points, labels)
    #num_samples = 1000
    num_batches = math.ceil(num_samples / float(FLAGS.batch_size))
    tf.logging.info('Eval num images %d', num_samples)
    tf.logging.info('Eval batch size %d and num batch %d',
                    FLAGS.batch_size, num_batches)
    # session_config = tf.ConfigProto(device_count={'GPU': 0})
    session_config = tf.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.2
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
          i = 0
          predictions_list = []
          labels_list = []
          while not coord.should_stop():
            logits_np, counts_logits_np, labels_np, counts_label = sess.run(
                [end_points['Logits_Predictions'], end_points['Counts_logits_Predictions'], labels, counts])
            logits_np = logits_np[0]
            labels_np = labels_np[0]
            counts_label = counts_label[0]
            counts_pre = np.argmax(counts_logits_np[0]) + 1
            predictions_np = np.zeros([28])
            labels_id = np.where(labels_np == 1)[0]
            predictions_id = list(np.argsort(logits_np)[(-counts_pre):])
            for idx in predictions_id:
              predictions_np[idx] = 1
            
            if FLAGS.threshould > 0:
              predictions_id = list(np.where(logits_np > (FLAGS.threshould/10000))[0])
              predictions_np = np.where(logits_np > (FLAGS.threshould/10000), 1, 0)
              if np.sum(predictions_np) == 0:
                max_id = np.argmax(logits_np)
                predictions_np[max_id] = 1
                predictions_id.append(max_id)

            labels_list.append(labels_np)
            predictions_list.append(predictions_np)
            i += 1
            sys.stdout.write('Image[{0}]--> labels:{1}, predictions: {2}\n'.format(i, labels_id, predictions_id))
            sys.stdout.flush()
        except tf.errors.OutOfRangeError:
          coord.request_stop()
          coord.join(threads)
        finally:
          sys.stdout.write('\n')
          sys.stdout.flush()
          all_labels = np.stack(labels_list, 0)
          all_pre = np.stack(predictions_list, 0)
          pred_rows = []
          for class_idx in range(28):
            class_labels = np.squeeze(all_labels[:, class_idx])
            class_pre = np.squeeze(all_pre[:, class_idx])
            class_f1_score = f1_score(class_labels, class_pre)
            class_precision_score = precision_score(class_labels, class_pre)
            class_recall_score = recall_score(class_labels, class_pre)
            sys.stdout.write('Class[{0}]--> F1_score: {1}\n'.format(class_idx, class_f1_score))
            sys.stdout.flush()
            pred_rows.append({'Class': str(class_idx),
                              'F1_score': str(class_f1_score),
                              'Precision': str(class_precision_score),
                              'Recall': str(class_recall_score),})
          all_f1_score = f1_score(all_labels, all_pre, average='macro')
          all_precision_score = precision_score(all_labels, all_pre, average='macro')
          all_recall_score = recall_score(all_labels, all_pre, average='macro')
          pred_rows.append({'Class': 'all',
                            'F1_score': str(all_f1_score),
                            'Precision': str(all_precision_score),
                            'Recall': str(all_recall_score),})
          sys.stdout.write('F1_score: {0}\n'.format(all_f1_score))
          sys.stdout.flush()
          submission_df = pd.DataFrame(pred_rows)[['Class', 'F1_score', 'Precision', 'Recall']]
          if FLAGS.threshould > 0:
            submission_df.to_csv(os.path.join('./result', 'protein_eval224_%04d.csv'%(int(FLAGS.threshould))), index=False)
          else:
            submission_df.to_csv(os.path.join('./result', 'protein_eval224_counts.csv'), index=False)


def main(unused_arg):
  eval_model()


if __name__ == '__main__':
  tf.app.run(main)
