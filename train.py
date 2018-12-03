from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.contrib import slim

import common
from net.mobilenet import mobilenet_v2
from dataset.get_dataset import get_dataset

flags = tf.app.flags

flags.DEFINE_string('master', '', 'Session master')
flags.DEFINE_integer('task', 0, 'Task')
flags.DEFINE_integer('ps_tasks', 0, 'Number of ps')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_integer('number_of_steps', None,
                     'Number of training steps to perform before stopping')
flags.DEFINE_integer('image_size', 96, 'Input image resolution')
flags.DEFINE_bool('quantize', False, 'Quantize training')
flags.DEFINE_string('fine_tune_checkpoint', '',
                    'Checkpoint from which to start finetuning.')
flags.DEFINE_string('train_logdir', './train_log',
                    'Directory for writing training checkpoints and logs')
flags.DEFINE_string('dataset_dir', '/media/jun/data/lcz/tfrecord', 'Location of dataset.')
#flags.DEFINE_string('dataset_dir', '/media/deeplearning/f3cff4c9-1ab9-47f0-8b82-231dedcbd61b/lcz/tfrecord/',
#                    'Location of dataset.')
flags.DEFINE_string('dataset', 'default', 'Name of the dataset.')
flags.DEFINE_string('train_split', 'train',
                    'Which split of the dataset to be used for training')
flags.DEFINE_integer('log_every_n_steps', 100, 'Number of steps per log')
flags.DEFINE_integer('save_summaries_secs', 100,
                     'How often to save summaries, secs')
flags.DEFINE_integer('save_interval_secs', 300,
                     'How often to save checkpoints, secs')

FLAGS = flags.FLAGS

_LEARNING_RATE_DECAY_FACTOR = 0.94

def get_learning_rate():
  if FLAGS.fine_tune_checkpoint:
    # If we are fine tuning a checkpoint we need to start at a lower learning
    # rate since we are farther along on training.
    return 1e-4
  else:
    return 0.01


def get_quant_delay():
  if FLAGS.fine_tune_checkpoint:
    # We can start quantizing immediately if we are finetuning.
    return 0
  else:
    # We need to wait for the model to train a bit before we quantize if we are
    # training from scratch.
    return 250000


def build_model():
  """Builds graph for model to train with rewrites for quantization.
  Returns:
    g: Graph with fake quantization ops and batch norm folding suitable for
    training quantized weights.
    train_tensor: Train op for execution during training.
  """
  g = tf.Graph()
  with g.as_default(), tf.device(
      tf.train.replica_device_setter(FLAGS.ps_tasks)):
    samples, num_samples = get_dataset(FLAGS.dataset, FLAGS.train_split, FLAGS.dataset_dir,
                                       FLAGS.image_size, FLAGS.batch_size, is_training=True)
    inputs = tf.identity(samples['data'], name='data')
    labels = tf.identity(samples['label'], name='label')
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=True, weight_decay=0.0001)):
      logits, _ = mobilenet_v2.mobilenet(
          inputs,
          is_training=True,
          depth_multiplier=FLAGS.depth_multiplier,
          num_classes=FLAGS.num_classes,
          finegrain_classification_mode=True)
    one_hot_labels = slim.one_hot_encoding(labels, FLAGS.num_classes, on_value=1.0, off_value=0.0)
    tf.losses.softmax_cross_entropy(one_hot_labels, logits)

    # Call rewriter to produce graph with fake quant ops and folded batch norms
    # quant_delay delays start of quantization till quant_delay steps, allowing
    # for better model accuracy.
    if FLAGS.quantize:
      tf.contrib.quantize.create_training_graph(quant_delay=get_quant_delay())
    
    # Gather update_ops
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # Configure the learning rate using an exponential decay.
    num_epochs_per_decay = 2.5
    decay_steps = int(num_samples / FLAGS.batch_size * num_epochs_per_decay)
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(
        get_learning_rate(),
        global_step,
        decay_steps,
        _LEARNING_RATE_DECAY_FACTOR,
        staircase=True)
    opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9)
    summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    total_losses = []
    softmax_cross_entropy_loss = tf.get_collection(tf.GraphKeys.LOSSES)
    softmax_cross_entropy_loss = tf.add_n(softmax_cross_entropy_loss,
                                          name='softmax_cross_entropy_loss')
    summaries.add(tf.summary.scalar('losses/softmax_cross_entropy_loss',
                                    softmax_cross_entropy_loss))
    total_losses.append(softmax_cross_entropy_loss)
    regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regularization_loss = tf.add_n(regularization_loss, name='regularization_loss')
    summaries.add(tf.summary.scalar('losses/regularization_loss', regularization_loss))
    total_losses.append(regularization_loss)

    total_loss = tf.add_n(total_losses, name='total_loss')
    grads_and_vars = opt.compute_gradients(total_loss)

    total_loss = tf.check_numerics(total_loss, 'LossTensor is inf or nan.')
    summaries.add(tf.summary.scalar('losses/total_loss', total_loss))

    grad_updates = opt.apply_gradients(grads_and_vars, global_step=global_step)
    update_ops.append(grad_updates)
    update_op = tf.group(*update_ops, name='update_barrier')
    with tf.control_dependencies([update_op]):
      train_tensor = tf.identity(total_loss, name='train_op')

  # Merge all summaries together.
  summary_op = tf.summary.merge(list(summaries))
  return g, train_tensor, summary_op


def get_checkpoint_init_fn():
  """Returns the checkpoint init_fn if the checkpoint is provided."""
  if FLAGS.fine_tune_checkpoint:
    variables_to_restore = slim.get_variables_to_restore(exclude=['MobilenetV2/Logits'])
    global_step_reset = tf.assign(tf.train.get_or_create_global_step(), 0)
    # When restoring from a floating point model, the min/max values for
    # quantized weights and activations are not present.
    # We instruct slim to ignore variables that are missing during restoration
    # by setting ignore_missing_vars=True
    slim_init_fn = slim.assign_from_checkpoint_fn(
        FLAGS.fine_tune_checkpoint,
        variables_to_restore,
        ignore_missing_vars=True)

    def init_fn(sess):
      slim_init_fn(sess)
      # If we are restoring from a floating point model, we need to initialize
      # the global step to zero for the exponential decay to result in
      # reasonable learning rates.
      sess.run(global_step_reset)
    return init_fn
  else:
    return None


def train_model():
  """Trains model."""
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.gfile.MakeDirs(FLAGS.train_logdir)
  tf.logging.info('Training on %s set', FLAGS.train_split)
  g, train_tensor, summary_op = build_model()
  # Soft placement allows placing on CPU ops without GPU implementation.
  gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
  session_config = tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
  with g.as_default():
    slim.learning.train(
        train_tensor,
        FLAGS.train_logdir,
        is_chief=(FLAGS.task == 0),
        master=FLAGS.master,
        log_every_n_steps=FLAGS.log_every_n_steps,
        graph=g,
        session_config=session_config,
        number_of_steps=FLAGS.number_of_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        init_fn=get_checkpoint_init_fn(),
        summary_op=summary_op,
        global_step=tf.train.get_global_step())


def main(unused_arg):
  train_model()


if __name__ == '__main__':
  tf.app.run(main)