from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.contrib import slim

import common, model
from utils import train_utils
from net.mobilenet import mobilenet_v2
#from dataset.get_lcz_dataset import get_dataset
from dataset import get_dataset
from utils import model_deploy

prefetch_queue = slim.prefetch_queue

flags = tf.app.flags
# Settings for multi-GPUs/multi-replicas training.

flags.DEFINE_integer('num_clones', 2, 'Number of clones to deploy.')
flags.DEFINE_boolean('clone_on_cpu', False, 'Use CPUs to deploy clones.')
flags.DEFINE_integer('num_replicas', 1, 'Number of worker replicas.')
flags.DEFINE_integer('startup_delay_steps', 15,
                     'Number of training steps between replicas startup.')
flags.DEFINE_integer('num_ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, then '
                     'the parameters are handled locally by the worker.')
flags.DEFINE_string('master', '', 'Session master')
flags.DEFINE_integer('task', 0, 'Task')
flags.DEFINE_integer('ps_tasks', 0, 'Number of ps')
flags.DEFINE_integer('train_batch_size', 2, 'Batch size')
flags.DEFINE_integer('number_of_steps', 2000000,
                     'Number of training steps to perform before stopping')
flags.DEFINE_integer('image_size', 112, 'Input image resolution')
flags.DEFINE_string('fine_tune_checkpoint', '',
                    'Checkpoint from which to start finetuning.')
# Set to False if one does not want to re-use the trained classifier weights.
flags.DEFINE_boolean('initialize_last_layer', False,
                     'Initialize the last layer.')
flags.DEFINE_float('last_layer_gradient_multiplier', 1.0,
                   'The gradient multiplier for last layers, which is used to '
                   'boost the gradient of last layers if the value > 1.')
flags.DEFINE_string('train_dir', '/mnt/home/hdd/hdd1/home/junq/lcz/train_log',
# flags.DEFINE_string('train_dir', '/home/jun/mynb/lcz/train_log',
                    'Directory for writing training checkpoints and logs')
# flags.DEFINE_string('dataset_dir', '/media/jun/data/tfrecord', 'Location of dataset.')
flags.DEFINE_string('dataset_dir', '/mnt/home/hdd/hdd1/home/junq/dataset', 'Location of dataset.')
# flags.DEFINE_string('dataset_dir', '/media/deeplearning/f3cff4c9-1ab9-47f0-8b82-231dedcbd61b/lcz/tfrecord/',
#                     'Location of dataset.')
flags.DEFINE_string('dataset', 'protein', 'Name of the dataset.')
flags.DEFINE_string('train_split', 'protein-01',
                    'Which split of the dataset to be used for training')
flags.DEFINE_integer('log_every_n_steps', 20, 'Number of steps per log')
flags.DEFINE_integer('save_summaries_secs', 60,
                     'How often to save summaries, secs')
flags.DEFINE_integer('save_interval_secs', 360,
                     'How often to save checkpoints, secs')
# Settings for training strategy.
flags.DEFINE_enum('learning_policy', 'step', ['poly', 'step'],
                  'Learning rate policy for training.')
flags.DEFINE_float('base_learning_rate', .001,
                   'The base learning rate for model training.')
flags.DEFINE_float('learning_rate_decay_factor', 0.98,
                   'The rate to decay the base learning rate.')
flags.DEFINE_integer('learning_rate_decay_step', 1000,
                     'Decay the base learning rate at a fixed step.')
flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')
flags.DEFINE_integer('slow_start_step', 0,
                     'Training model with small learning rate for few steps.')
flags.DEFINE_float('slow_start_learning_rate', 1e-4,
                   'Learning rate employed during slow start.')
flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')
# For weight_decay, use 0.00004 for MobileNet-V2 or Xcpetion model variants.
# Use 0.0001 for ResNet model variants.
flags.DEFINE_float('weight_decay', 0.0001,
                   'The value of the weight decay for training.')
# Set to True if one wants to fine-tune the batch norm parameters in DeepLabv3.
# Set to False and use small batch size to save GPU memory.
flags.DEFINE_boolean('fine_tune_batch_norm', True,
                     'Fine tune the batch norm parameters or not.')
flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')
flags.DEFINE_boolean('save_summaries_variables', False,
                     'Save the variables summaries or not.')

FLAGS = flags.FLAGS


def _build_model(inputs_queue, clone_batch_size):
  """Builds a clone of train model.

  Args:
    inputs_queue: A prefetch queue for images and labels.
  Returns:
    A dictionary of logits names to logits.
  """
  samples = inputs_queue.dequeue()
  batch_size = clone_batch_size * (FLAGS.num_classes - 1)
  inputs = tf.identity(samples['image'], name='image')
  labels = tf.identity(tf.concat([tf.zeros([batch_size, 1]), samples['label']], -1), name='label')
  model_options = common.ModelOptions(output_stride=FLAGS.output_stride)
  net, end_points = model.get_features(
      inputs,
      model_options=model_options,
      weight_decay=FLAGS.weight_decay,
      is_training=True,
      fine_tune_batch_norm=FLAGS.fine_tune_batch_norm)
  logits, _ = model.classification(net, end_points, 
                                   num_classes=FLAGS.num_classes,
                                   is_training=True)
  if FLAGS.multi_label:
    with tf.name_scope('Multilabel_logits'):
      logits = slim.softmax(logits)
      half_batch_size = batch_size / 2
      train_utils.focal_loss(labels[:, 0], logits[:, 0], scope='class_loss_00')
      for i in range(1, FLAGS.num_classes):
        class_logits = tf.identity(logits[:, i], name='class_logits_%02d'%(i))
        class_labels = tf.identity(labels[:, i], name='class_labels_%02d'%(i))
        num_positive = tf.reduce_sum(class_labels)
        num_negative = batch_size - num_positive
        weights = tf.where(tf.equal(class_labels, 1.0),
                           tf.tile([half_batch_size/num_positive], [batch_size]),
                           tf.tile([half_batch_size/num_negative], [batch_size]))
        train_utils.focal_loss(class_labels, class_logits,
                               weights=weights, scope='class_loss_%02d'%(i))
  else:
    logits = slim.softmax(logits)
    train_utils.focal_loss(labels, logits, scope='cls_loss')

  if (FLAGS.dataset == 'protein') and FLAGS.add_counts_logits:
    counts = tf.identity(samples['counts'], name='counts')
    one_hot_counts = slim.one_hot_encoding(counts, 6)
    counts_logits, _ = model.classification(net, end_points, num_classes=6,
                                            is_training=True, scope='Counts_logits')
    counts_logits = slim.softmax(counts_logits)
    train_utils.focal_loss(one_hot_counts, counts_logits, scope='counts_loss')
    return logits, counts_logits
  return logits


def main(unused_arg):
  tf.logging.set_verbosity(tf.logging.INFO)
  # Set up deployment (i.e., multi-GPUs and/or multi-replicas).
  config = model_deploy.DeploymentConfig(
      num_clones=FLAGS.num_clones,
      clone_on_cpu=FLAGS.clone_on_cpu,
      replica_id=FLAGS.task,
      num_replicas=FLAGS.num_replicas,
      num_ps_tasks=FLAGS.num_ps_tasks)

  # Split the batch across GPUs.
  assert FLAGS.train_batch_size % config.num_clones == 0, (
      'Training batch size not divisble by number of clones (GPUs).')

  clone_batch_size = FLAGS.train_batch_size // config.num_clones

  tf.gfile.MakeDirs(FLAGS.train_dir)
  tf.logging.info('Training on %s set', FLAGS.train_split)

  with tf.Graph().as_default() as graph:
    with tf.device(config.inputs_device()):
      samples, _ = get_dataset.get_dataset(FLAGS.dataset, FLAGS.dataset_dir,
                                           split_name=FLAGS.train_split,
                                           is_training=True,
                                           image_size=[FLAGS.image_size, FLAGS.image_size],
                                           batch_size=clone_batch_size,
                                           channel=FLAGS.input_channel)
      inputs_queue = prefetch_queue.prefetch_queue(
          samples, capacity=128 * config.num_clones)
      # Create the global step on the device storing the variables.
    with tf.device(config.variables_device()):
      global_step = tf.train.get_or_create_global_step()
      # Define the model and create clones.
      model_fn = _build_model
      model_args = (inputs_queue, clone_batch_size)
      clones = model_deploy.create_clones(config, model_fn, args=model_args)

      # Gather update_ops from the first clone. These contain, for example,
      # the updates for the batch_norm variables created by model_fn.
      first_clone_scope = config.clone_scope(0)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)
    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    # Add summaries for model variables.
    if FLAGS.save_summaries_variables:
      for model_var in slim.get_model_variables():
        summaries.add(tf.summary.histogram(model_var.op.name, model_var))

    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))
    # Build the optimizer based on the device specification.
    with tf.device(config.optimizer_device()):
      learning_rate = train_utils.get_model_learning_rate(
          FLAGS.learning_policy, FLAGS.base_learning_rate,
          FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
          FLAGS.number_of_steps, FLAGS.learning_power,
          FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
      optimizer = tf.train.AdamOptimizer(learning_rate)
      #optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=FLAGS.momentum)
      summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    startup_delay_steps = FLAGS.task * FLAGS.startup_delay_steps
    with tf.device(config.variables_device()):
      total_loss, grads_and_vars = model_deploy.optimize_clones(
          clones, optimizer)
      total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan.')
      summaries.add(tf.summary.scalar('total_loss', total_loss))

      # Modify the gradients for biases and last layer variables.
      if (FLAGS.dataset == 'protein') and FLAGS.add_counts_logits:
        last_layers = ['Logits', 'Counts_logits']
      else:
        last_layers = ['Logits']
      grad_mult = train_utils.get_model_gradient_multipliers(
          last_layers, FLAGS.last_layer_gradient_multiplier)
      if grad_mult:
        grads_and_vars = slim.learning.multiply_gradients(
            grads_and_vars, grad_mult)

      # Create gradient update op.
      grad_updates = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
      update_ops.append(grad_updates)
      update_op = tf.group(*update_ops)
      with tf.control_dependencies([update_op]):
        train_tensor = tf.identity(total_loss, name='train_op')

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(
        tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries))

    # Soft placement allows placing on CPU ops without GPU implementation.
    session_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.8

    # Start the training.
    slim.learning.train(
        train_tensor,
        FLAGS.train_dir,
        is_chief=(FLAGS.task == 0),
        master=FLAGS.master,
        graph=graph,
        log_every_n_steps=FLAGS.log_every_n_steps,
        session_config=session_config,
        startup_delay_steps=startup_delay_steps,
        number_of_steps=FLAGS.number_of_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        init_fn=train_utils.get_model_init_fn(
            FLAGS.train_dir,
            FLAGS.fine_tune_checkpoint,
            FLAGS.initialize_last_layer,
            last_layers,
            ignore_missing_vars=True),
        summary_op=summary_op,
        saver=tf.train.Saver(max_to_keep=50))


if __name__ == '__main__':
  flags.mark_flag_as_required('train_dir')
  flags.mark_flag_as_required('fine_tune_checkpoint')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()
