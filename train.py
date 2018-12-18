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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

flags = tf.app.flags

flags.DEFINE_string('master', '', 'Session master')
flags.DEFINE_integer('task', 0, 'Task')
flags.DEFINE_integer('ps_tasks', 0, 'Number of ps')
flags.DEFINE_integer('batch_size', 4, 'Batch size')
flags.DEFINE_integer('number_of_steps', 1500000,
                     'Number of training steps to perform before stopping')
flags.DEFINE_integer('image_size', 112, 'Input image resolution')
flags.DEFINE_string('fine_tune_checkpoint', '',
                    'Checkpoint from which to start finetuning.')
flags.DEFINE_string('train_dir', '/mnt/home/hdd/hdd1/home/junq/lcz/train_log',
                    'Directory for writing training checkpoints and logs')
flags.DEFINE_string('dataset_dir', '/mnt/home/hdd/hdd1/home/junq/dataset', 'Location of dataset.')
#flags.DEFINE_string('dataset_dir', '/media/deeplearning/f3cff4c9-1ab9-47f0-8b82-231dedcbd61b/lcz/tfrecord/',
#                    'Location of dataset.')
flags.DEFINE_string('dataset', 'protein', 'Name of the dataset.')
flags.DEFINE_string('train_split', 'protein-02',
                    'Which split of the dataset to be used for training')
flags.DEFINE_integer('log_every_n_steps', 20, 'Number of steps per log')
flags.DEFINE_integer('save_summaries_secs', 60,
                     'How often to save summaries, secs')
flags.DEFINE_integer('save_interval_secs', 360,
                     'How often to save checkpoints, secs')
# Settings for training strategy.
flags.DEFINE_enum('learning_policy', 'step', ['poly', 'step'],
                  'Learning rate policy for training.')
# Use 0.007 when training on PASCAL augmented training set, train_aug. When
# fine-tuning on PASCAL trainval set, use learning rate=0.0001.
flags.DEFINE_float('base_learning_rate', .0001,
                   'The base learning rate for model training.')
flags.DEFINE_float('learning_rate_decay_factor', 0.96,
                   'The rate to decay the base learning rate.')
flags.DEFINE_integer('learning_rate_decay_step', 2500,
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
flags.DEFINE_float('weight_decay', 0.0004,
                   'The value of the weight decay for training.')
# Set to True if one wants to fine-tune the batch norm parameters in DeepLabv3.
# Set to False and use small batch size to save GPU memory.
flags.DEFINE_boolean('fine_tune_batch_norm', True,
                     'Fine tune the batch norm parameters or not.')
flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

FLAGS = flags.FLAGS

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
    samples, _ = get_dataset.get_dataset(FLAGS.dataset, FLAGS.dataset_dir,
                                         split_name=FLAGS.train_split,
                                         is_training=True,
                                         image_size=[FLAGS.image_size, FLAGS.image_size],
                                         batch_size=FLAGS.batch_size,
                                         channel=FLAGS.input_channel)
    inputs = tf.identity(samples['image'], name='image')
    labels = tf.identity(samples['label'], name='label')
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
    tf.losses.softmax_cross_entropy(labels, logits)

    # Gather update_ops
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    global_step = tf.train.get_or_create_global_step()
    learning_rate = train_utils.get_model_learning_rate(
          FLAGS.learning_policy, FLAGS.base_learning_rate,
          FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
          FLAGS.number_of_steps, FLAGS.learning_power,
          FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)
    #opt = tf.train.RMSPropOptimizer(learning_rate, momentum=FLAGS.momentum)
    summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    cls_loss = tf.get_collection(tf.GraphKeys.LOSSES)
    for loss in cls_loss:
      summaries.add(tf.summary.scalar('losses/%s'%(loss.op.name), loss))
    cls_loss = tf.add_n(cls_loss, name='cls_loss')
    regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regularization_loss = tf.add_n(regularization_loss, name='regularization_loss')
    summaries.add(tf.summary.scalar('losses/regularization_loss', regularization_loss))

    total_loss = tf.add(cls_loss, regularization_loss, name='total_loss')
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
  tf.gfile.MakeDirs(FLAGS.train_dir)
  tf.logging.info('Training on %s set', FLAGS.train_split)
  g, train_tensor, summary_op = build_model()
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction = 0.7
  with g.as_default():
    slim.learning.train(
        train_tensor,
        FLAGS.train_dir,
        is_chief=(FLAGS.task == 0),
        master=FLAGS.master,
        log_every_n_steps=FLAGS.log_every_n_steps,
        session_config=config,
        graph=g,
        number_of_steps=FLAGS.number_of_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        init_fn=get_checkpoint_init_fn(),
        summary_op=summary_op,
        global_step=tf.train.get_global_step(),
        saver=tf.train.Saver(max_to_keep=50))


def main(unused_arg):
  train_model()


if __name__ == '__main__':
  tf.app.run(main)
