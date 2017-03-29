import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn import BasicLSTMCell, GRUCell, MultiRNNCell
from tensorflow.python.util import nest


def multi_bidirectional_rnn(cells, inputs, sequence_length=None, dtype=None, parallel_iterations=None,
                            swap_memory=False, time_major=False, trainable_initial_state=True, **kwargs):
    if not time_major:
        time_dim = 1
        batch_dim = 0
    else:
        time_dim = 0
        batch_dim = 1

    batch_size = tf.shape(inputs)[batch_dim]

    output_states_fw = []
    output_states_bw = []
    for i, (cell_fw, cell_bw) in enumerate(cells):
        # forward direction
        with tf.variable_scope('forward_{}'.format(i + 1)) as fw_scope:
            if trainable_initial_state:
                initial_state = get_variable_unsafe('initial_state', initializer=tf.zeros([cell_fw.state_size]),
                                                    dtype=dtype)
                initial_state = tf.reshape(tf.tile(initial_state, [batch_size]),
                                           shape=[batch_size, cell_fw.state_size])
            else:
                initial_state = None

            inputs_fw, output_state_fw = rnn.dynamic_rnn(
                cell=cell_fw, inputs=inputs, sequence_length=sequence_length, initial_state=initial_state,
                dtype=dtype, parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                time_major=time_major, scope=fw_scope
            )

        # backward direction
        inputs_reversed = tf.reverse_sequence(
            input=inputs, seq_lengths=sequence_length, seq_dim=time_dim, batch_dim=batch_dim
        )

        with tf.variable_scope('backward_{}'.format(i + 1)) as bw_scope:
            if trainable_initial_state:
                initial_state = get_variable_unsafe('initial_state', initializer=tf.zeros([cell_bw.state_size]),
                                                    dtype=dtype)
                initial_state = tf.reshape(tf.tile(initial_state, [batch_size]),
                                           shape=[batch_size, cell_bw.state_size])
            else:
                initial_state = None

            inputs_bw, output_state_bw = rnn.dynamic_rnn(
                cell=cell_bw, inputs=inputs_reversed, sequence_length=sequence_length, initial_state=initial_state,
                dtype=dtype, parallel_iterations=parallel_iterations, swap_memory=swap_memory, time_major=time_major,
                scope=bw_scope
            )

        inputs_bw_reversed = tf.reverse_sequence(
            input=inputs_bw, seq_lengths=sequence_length,
            seq_dim=time_dim, batch_dim=batch_dim
        )
        inputs = tf.concat([inputs_fw, inputs_bw_reversed], 2)
        output_states_fw.append(output_state_fw)
        output_states_bw.append(output_state_bw)

    return inputs, tf.concat(output_states_fw, 1), tf.concat(output_states_bw, 1)


def multi_rnn(cells, inputs, sequence_length=None, dtype=None, parallel_iterations=None, swap_memory=False,
              time_major=False, trainable_initial_state=True, **kwargs):
    batch_size = tf.shape(inputs)[0]     # TODO: Fix time major stuff

    output_states = []
    for i, cell in enumerate(cells):
        with tf.variable_scope('forward_{}'.format(i + 1)) as scope:
            if trainable_initial_state:
                initial_state = get_variable_unsafe('initial_state', initializer=tf.zeros([cell.state_size]),
                                                    dtype=dtype)
                initial_state = tf.reshape(tf.tile(initial_state, [batch_size]),
                                           shape=[batch_size, cell.state_size])
            else:
                initial_state = None

            inputs, output_state = rnn.dynamic_rnn(
                cell=cell, inputs=inputs, sequence_length=sequence_length, initial_state=initial_state, dtype=dtype,
                parallel_iterations=parallel_iterations, swap_memory=swap_memory, time_major=time_major, scope=scope
            )

        output_states.append(output_state)

    return inputs, tf.concat(output_states, 1)


def unsafe_decorator(fun):
    """
    Wrapper that automatically handles the `reuse' parameter.
    This is rather risky, as it can lead to reusing variables
    by mistake.
    """
    def fun_(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except ValueError as e:
            if 'reuse' in str(e):
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    return fun(*args, **kwargs)
            else:
                raise e

    return fun_


def linear(args, output_size, bias, bias_start=0.0, scope=None, initializer=None):
    """
    Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Same as as `tf.nn.rnn_cell._linear`, with the addition of an `initializer` parameter.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".
      initializer: used to initialize W

    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """

    if not nest.is_sequence(args):
        args = [args]

    # calculate the total size of arguments on dimension 1
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if shape[1] is None:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size], dtype=dtype,
                                 initializer=initializer)
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(args, 1), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable("Bias", [output_size], dtype=dtype,
                                    initializer=tf.constant_initializer(bias_start, dtype=dtype))
    return res + bias_term


get_variable_unsafe = unsafe_decorator(tf.get_variable)
GRUCell_unsafe = unsafe_decorator(GRUCell)
BasicLSTMCell_unsafe = unsafe_decorator(BasicLSTMCell)
MultiRNNCell_unsafe = unsafe_decorator(MultiRNNCell)
linear_unsafe = unsafe_decorator(linear)
multi_rnn_unsafe = unsafe_decorator(multi_rnn)
multi_bidirectional_rnn_unsafe = unsafe_decorator(multi_bidirectional_rnn)
