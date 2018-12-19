import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops

def metric_variable(shape, dtype, validate_shape=True, name=None):
    """Create variable in `GraphKeys.(LOCAL|METRIC_VARIABLES`) collections."""

    return variable_scope.variable(
        lambda: array_ops.zeros(shape, dtype),
        trainable=False,
        collections=[ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.METRIC_VARIABLES],
        validate_shape=validate_shape,
        name=name,
    )


def streaming_counts(y_true, y_pred, num_classes):
    # Weights for the weighted f1 score
    weights = metric_variable(
        shape=[num_classes], dtype=tf.int64, validate_shape=False, name="weights"
    )
    # Counts for the macro f1 score
    tp_mac = metric_variable(
        shape=[num_classes], dtype=tf.int64, validate_shape=False, name="tp_mac"
    )
    fp_mac = metric_variable(
        shape=[num_classes], dtype=tf.int64, validate_shape=False, name="fp_mac"
    )
    fn_mac = metric_variable(
        shape=[num_classes], dtype=tf.int64, validate_shape=False, name="fn_mac"
    )
    # Counts for the micro f1 score
    tp_mic = metric_variable(
        shape=[], dtype=tf.int64, validate_shape=False, name="tp_mic"
    )
    fp_mic = metric_variable(
        shape=[], dtype=tf.int64, validate_shape=False, name="fp_mic"
    )
    fn_mic = metric_variable(
        shape=[], dtype=tf.int64, validate_shape=False, name="fn_mic"
    )

    # Update ops, as in the previous section:
    #   - Update ops for the macro f1 score
    up_tp_mac = tf.assign_add(tp_mac, tf.count_nonzero(y_pred * y_true, axis=0))
    up_fp_mac = tf.assign_add(fp_mac, tf.count_nonzero(y_pred * (y_true - 1), axis=0))
    up_fn_mac = tf.assign_add(fn_mac, tf.count_nonzero((y_pred - 1) * y_true, axis=0))

    #   - Update ops for the micro f1 score
    up_tp_mic = tf.assign_add(
        tp_mic, tf.count_nonzero(y_pred * y_true, axis=None)
    )
    up_fp_mic = tf.assign_add(
        fp_mic, tf.count_nonzero(y_pred * (y_true - 1), axis=None)
    )
    up_fn_mic = tf.assign_add(
        fn_mic, tf.count_nonzero((y_pred - 1) * y_true, axis=None)
    )
    # Update op for the weights, just summing
    up_weights = tf.assign_add(weights, tf.reduce_sum(y_true, axis=0))

    # Grouping values
    counts = (tp_mac, fp_mac, fn_mac, tp_mic, fp_mic, fn_mic, weights)
    updates = tf.group(up_tp_mic, up_fp_mic, up_fn_mic, up_tp_mac, up_fp_mac, up_fn_mac, up_weights)

    return counts, updates


def streaming_f1(counts):
    # unpacking values
    tp_mac, fp_mac, fn_mac, tp_mic, fp_mic, fn_mic, weights = counts

    # normalize weights
    weights /= tf.reduce_sum(weights)

    # computing the micro f1 score
    prec_mic = tp_mic / (tp_mic + fp_mic)
    rec_mic = tp_mic / (tp_mic + fn_mic)
    f1_mic = 2 * prec_mic * rec_mic / (prec_mic + rec_mic)
    f1_mic = tf.reduce_mean(f1_mic)

    # computing the macro and wieghted f1 score
    prec_mac = tp_mac / (tp_mac + fp_mac)
    rec_mac = tp_mac / (tp_mac + fn_mac)
    f1_mac = 2 * prec_mac * rec_mac / (prec_mac + rec_mac)
    f1_wei = tf.reduce_sum(f1_mac * weights)
    f1_mac = tf.reduce_mean(f1_mac)

    return f1_mic, f1_mac, f1_wei