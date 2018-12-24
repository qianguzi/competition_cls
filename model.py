import tensorflow as tf
from tensorflow.contrib import slim

from net import feature_extractor

def get_features(images,
                 model_options,
                 weight_decay=0.0001,
                 is_training=False,
                 fine_tune_batch_norm=False):
  net, end_points = feature_extractor.extract_features(
      images,
      output_stride=model_options.output_stride,
      multi_grid=model_options.multi_grid,
      model_variant=model_options.model_variant,
      depth_multiplier=model_options.depth_multiplier,
      weight_decay=weight_decay,
      reuse=tf.AUTO_REUSE,
      is_training=is_training,
      fine_tune_batch_norm=fine_tune_batch_norm)
  
  net = tf.identity(net, name='embedding')
  return net, end_points


def classification(net, end_points, 
                   num_classes=18, 
                   is_training=False, 
                   prediction_fn=slim.softmax,
                   scope=None):
  with tf.variable_scope(scope, 'Logits'):
    net = global_pool(net)
    end_points['global_pool'] = net
    if not num_classes:
       return net, end_points
    net = slim.dropout(net, scope='Dropout', is_training=is_training)
    # 1 x 1 x num_classes
    # Note: legacy scope name.
    logits = slim.conv2d(
        net,
        num_classes, [1, 1],
        activation_fn=None,
        normalizer_fn=None,
        biases_initializer=tf.zeros_initializer(),
        scope='Conv2d_1c_1x1')
    logits = tf.squeeze(logits, [1, 2])

    logits = tf.identity(logits, name='output')
    end_points['Logits'] = logits
    if prediction_fn:
      end_points['Predictions'] = prediction_fn(logits, 'Predictions')
  return logits, end_points


def hierarchical_classification(net, end_points, is_training=False):
  with tf.variable_scope('Hierarchical_logits'):
    net = global_pool(net)
    end_points['global_pool'] = net
    net = slim.dropout(net, scope='Dropout', is_training=is_training)
    low_logits = slim.conv2d(
        net,
        2, [1, 1],
        activation_fn=None,
        normalizer_fn=None,
        biases_initializer=tf.zeros_initializer())
    low_logits = tf.squeeze(low_logits, [1, 2])
    end_points['Low_logits'] = low_logits

    low_prediction = slim.softmax(low_logits, 'softmax_low')
    end_points['Prediction_low'] = low_prediction

    high_logits = slim.conv2d(
        net,
        18, [1, 1],
        activation_fn=None,
        normalizer_fn=None,
        biases_initializer=tf.zeros_initializer())
    high_logits = tf.squeeze(high_logits, [1, 2])
    end_points['High_logits'] = high_logits
    high_prediction = slim.softmax(high_logits, 'softmax_high')
    end_points['Prediction_high'] = high_prediction
    a_prediction = tf.multiply(tf.expand_dims(low_prediction[:,0], -1), high_prediction[:,1:11])
    b_prediction = tf.multiply(tf.expand_dims(low_prediction[:,1], -1), high_prediction[:,11:])
    prediction = tf.concat([tf.expand_dims(high_prediction[:,0], -1), a_prediction, b_prediction],
                           -1, name='Predictions')
    end_points['Predictions'] = prediction
  return end_points


def global_pool(input_tensor, pool_op=tf.nn.avg_pool):
  """Applies avg pool to produce 1x1 output.

  NOTE: This function is funcitonally equivalenet to reduce_mean, but it has
  baked in average pool which has better support across hardware.

  Args:
    input_tensor: input tensor
    pool_op: pooling op (avg pool is default)
  Returns:
    a tensor batch_size x 1 x 1 x depth.
  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size = tf.convert_to_tensor(
        [1, tf.shape(input_tensor)[1],
         tf.shape(input_tensor)[2], 1])
  else:
    kernel_size = [1, shape[1], shape[2], 1]
  output = pool_op(
      input_tensor, ksize=kernel_size, strides=[1, 1, 1, 1], padding='VALID')
  # Recover output shape, for unknown shape.
  output.set_shape([None, 1, 1, None])
  return output