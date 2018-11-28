import tensorflow as tf

from mobilenet import mobilenet_v2
from dataset import preprocess

slim = tf.contrib.slim

flags = tf.app.flags

flags.DEFINE_string('master', '', 'Session master')
flags.DEFINE_integer('task', 0, 'Task')
flags.DEFINE_integer('ps_tasks', 0, 'Number of ps')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_integer('num_classes', 18, 'Number of classes to distinguish')
flags.DEFINE_integer('number_of_steps', None,
                     'Number of training steps to perform before stopping')
flags.DEFINE_integer('image_size', 32, 'Input image resolution')
flags.DEFINE_float('depth_multiplier', 1.5, 'Depth multiplier for mobilenet')
flags.DEFINE_bool('quantize', False, 'Quantize training')
flags.DEFINE_string('fine_tune_checkpoint', '',
                    'Checkpoint from which to start finetuning.')
flags.DEFINE_string('checkpoint_dir', './train_log',
                    'Directory for writing training checkpoints and logs')
flags.DEFINE_string('dataset_dir', '/media/jun/data/lcz/tfrecord/train-*', 'Location of dataset')
flags.DEFINE_integer('log_every_n_steps', 100, 'Number of steps per log')
flags.DEFINE_integer('save_summaries_secs', 100,
                     'How often to save summaries, secs')
flags.DEFINE_integer('save_interval_secs', 100,
                     'How often to save checkpoints, secs')

FLAGS = flags.FLAGS

_LEARNING_RATE_DECAY_FACTOR = 0.94
_DATASET_SIZE = 121992
#_DATASET_SIZE_VAL = 24119

def get_learning_rate():
  if FLAGS.fine_tune_checkpoint:
    # If we are fine tuning a checkpoint we need to start at a lower learning
    # rate since we are farther along on training.
    return 1e-4
  else:
    return 0.045


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
    samples = preprocess.get_batch(FLAGS.dataset_dir, FLAGS.batch_size, is_training=True)
    inputs = tf.identity(samples['data'], name='data')
    labels = tf.identity(samples['label'], name='label')
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=True)):
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

    total_loss = tf.losses.get_total_loss(name='total_loss')
    # Configure the learning rate using an exponential decay.
    num_epochs_per_decay = 2.5
    decay_steps = int(_DATASET_SIZE / FLAGS.batch_size * num_epochs_per_decay)

    learning_rate = tf.train.exponential_decay(
        get_learning_rate(),
        tf.train.get_or_create_global_step(),
        decay_steps,
        _LEARNING_RATE_DECAY_FACTOR,
        staircase=True)
    opt = tf.train.GradientDescentOptimizer(learning_rate)

    train_tensor = slim.learning.create_train_op(
        total_loss,
        optimizer=opt)

  slim.summaries.add_scalar_summary(total_loss, 'total_loss', 'losses')
  slim.summaries.add_scalar_summary(learning_rate, 'learning_rate', 'training')
  return g, train_tensor


def get_checkpoint_init_fn():
  """Returns the checkpoint init_fn if the checkpoint is provided."""
  if FLAGS.fine_tune_checkpoint:
    variables_to_restore = slim.get_variables_to_restore()
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
  """Trains mobilenet_v1."""
  g, train_tensor = build_model()
  with g.as_default():
    slim.learning.train(
        train_tensor,
        FLAGS.checkpoint_dir,
        is_chief=(FLAGS.task == 0),
        master=FLAGS.master,
        log_every_n_steps=FLAGS.log_every_n_steps,
        graph=g,
        number_of_steps=FLAGS.number_of_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        init_fn=get_checkpoint_init_fn(),
        global_step=tf.train.get_global_step())


def main(unused_arg):
  train_model()


if __name__ == '__main__':
  tf.app.run(main)