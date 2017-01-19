import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell


def multi_bidirectional_rnn(cells, inputs, sequence_length=None, dtype=None, parallel_iterations=None,
                            swap_memory=False, time_major=False, time_pooling=None, pooling_avg=None,
                            residual_connections=False, trainable_initial_state=True, **kwargs):
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
        new_inputs = tf.concat(2, [inputs_fw, inputs_bw_reversed])

        if residual_connections and i < len(cells) - 1:
            # the output's dimension is twice that of the initial input (because of bidir)
            if i == 0:
                inputs = tf.tile(inputs, (1, 1, 2))  # FIXME: temporary solution
            inputs = new_inputs + inputs
        else:
            inputs = new_inputs

        if time_pooling and i < len(cells) - 1:
            inputs, sequence_length = apply_time_pooling(inputs, sequence_length, time_pooling[i], pooling_avg)

        output_states_fw.append(output_state_fw)
        output_states_bw.append(output_state_bw)

    return inputs, tf.concat(1, output_states_fw), tf.concat(1, output_states_bw)


def multi_rnn(cells, inputs, sequence_length=None, dtype=None, parallel_iterations=None, swap_memory=False,
              time_major=False, time_pooling=None, pooling_avg=None, residual_connections=False,
              trainable_initial_state=True, **kwargs):
    assert time_pooling is None or len(time_pooling) == len(cells) - 1

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

            new_inputs, output_state = rnn.dynamic_rnn(
                cell=cell, inputs=inputs, sequence_length=sequence_length, initial_state=initial_state, dtype=dtype,
                parallel_iterations=parallel_iterations, swap_memory=swap_memory, time_major=time_major, scope=scope
            )

        if residual_connections and i < len(cells) - 1:
            inputs = new_inputs + inputs
        else:
            inputs = new_inputs

        if time_pooling and i < len(cells) - 1:
            inputs, sequence_length = apply_time_pooling(inputs, sequence_length, time_pooling[i], pooling_avg)

        output_states.append(output_state)

    return inputs, tf.concat(1, output_states)


def apply_time_pooling(inputs, sequence_length, stride, pooling_avg=False):
    shape = [tf.shape(inputs)[0], tf.shape(inputs)[1], inputs.get_shape()[2].value]

    if pooling_avg:
        inputs_ = [inputs[:, i::stride, :] for i in range(stride)]

        max_len = tf.shape(inputs_[0])[1]
        for k in range(1, stride):
            len_ = tf.shape(inputs_[k])[1]
            paddings = tf.pack([[0, 0], [0, max_len - len_], [0, 0]])
            inputs_[k] = tf.pad(inputs_[k], paddings=paddings)

        inputs = tf.reduce_sum(inputs_, 0) / len(inputs_)
    else:
        inputs = inputs[:, ::stride, :]

    inputs = tf.reshape(inputs, tf.pack([shape[0], tf.shape(inputs)[1], shape[2]]))
    sequence_length = (sequence_length + stride - 1) // stride  # rounding up

    return inputs, sequence_length


def unsafe_decorator(fun):
    """
    Wrapper that automatically handles the `reuse' parameter.
    This is rather unsafe, as it can lead to reusing variables
    by mistake, without knowing about it.
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


class MultiRNNCell(rnn_cell.RNNCell):
    """
    Same as rnn_cell.MultiRNNCell, except it accepts an additional `residual_connections` parameter
    """

    def __init__(self, cells, state_is_tuple=False, residual_connections=False):
        self._cells = cells
        self._state_is_tuple = state_is_tuple
        self._residual_connections = residual_connections

    @property
    def state_size(self):
        if self._state_is_tuple:
            return tuple(cell.state_size for cell in self._cells)
        else:
            return sum([cell.state_size for cell in self._cells])

    @property
    def output_size(self):
        return self._cells[-1].output_size

    def __call__(self, inputs, state, scope=None):
        cur_state_pos = 0
        cur_inp = inputs
        new_states = []
        for i, cell in enumerate(self._cells):
            with tf.variable_scope('cell_{}'.format(i + 1)):
                if self._state_is_tuple:
                    cur_state = state[i]
                else:
                    cur_state = tf.slice(state, [0, cur_state_pos], [-1, cell.state_size])
                    cur_state_pos += cell.state_size
                new_inp, new_state = cell(cur_inp, cur_state)
                if self._residual_connections and i < len(self._cells) - 1:
                    cur_inp = cur_inp + new_inp
                else:
                    cur_inp = new_inp
                new_states.append(new_state)
        new_states = (tuple(new_states) if self._state_is_tuple else tf.concat(1, new_states))
        return cur_inp, new_states


class GRUCell(rnn_cell.RNNCell):
    def __init__(self, num_units, activation=tf.nn.tanh, initializer=None):
        self._num_units = num_units
        self._activation = activation
        self._initializer = initializer

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # we start with bias of 1.0 to not reset and not update
            state_to_gates = linear(state, self._num_units * 2, False, scope='state_to_gates',
                                    initializer=self._initializer)
            input_to_gates = linear(inputs, self._num_units * 2, True, scope='input_to_gates')

            gates = tf.nn.sigmoid(state_to_gates + input_to_gates)
            update = gates[:, :self._num_units]
            reset = gates[:, self._num_units:]

            state_to_state = linear(state, self._num_units, False, scope='state_to_state',
                                    initializer=self._initializer)
            input_to_state = linear(inputs, self._num_units, True, scope='input_to_state')
            new_state = self._activation(reset * state_to_state + input_to_state)

            new_state = update * new_state + (1 - update) * state
        return new_state, new_state


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
  if args is None or (tf.nn.nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not tf.nn.nest.is_sequence(args):
    args = [args]

  # calculate the total size of arguments on dimension 1
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
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
      res = tf.matmul(tf.concat(1, args), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable("Bias", [output_size], dtype=dtype,
                                initializer=tf.constant_initializer(bias_start, dtype=dtype))
  return res + bias_term


def orthogonal_initializer(scale=1.0, dtype=tf.float32):
    """Initialize a random orthogonal matrix.

    Only works for 2D arrays.

    Parameters
    ----------
    scale : float, optional
        Multiply the resulting matrix with a scalar. Defaults to 1.
        For a discussion of the importance of scale for training time
        and generalization refer to [Saxe2013]_.

        .. [Saxe2013] Saxe, A.M., McClelland, J.L., Ganguli, S., 2013.,
           *Exact solutions to the nonlinear dynamics of learning in deep
           linear neural networks*,
           arXiv:1312.6120 [cond-mat, q-bio, stat].

    """

    def _initializer(shape, dtype=dtype, partition_info=None):
        if len(shape) != 2:
            raise ValueError

        if shape[0] == shape[1]:
            # For square weight matrices we can simplify the logic
            # and be more exact:
            M = np.random.randn(*shape)
            Q, R = np.linalg.qr(M)
            Q = Q * np.sign(np.diag(R))
            return Q * scale

        M1 = np.random.randn(shape[0], shape[0])
        M2 = np.random.randn(shape[1], shape[1])

        # QR decomposition of matrix with entries in N(0, 1) is random
        Q1, R1 = np.linalg.qr(M1)
        Q2, R2 = np.linalg.qr(M2)
        # Correct that NumPy doesn't force diagonal of R to be non-negative
        Q1 = Q1 * np.sign(np.diag(R1))
        Q2 = Q2 * np.sign(np.diag(R2))

        n_min = min(shape[0], shape[1])
        return tf.constant(np.dot(Q1[:, :n_min], Q2[:n_min, :]) * scale, dtype=dtype)
    return _initializer


get_variable_unsafe = unsafe_decorator(tf.get_variable)
GRUCell_unsafe = unsafe_decorator(rnn_cell.GRUCell)
BasicLSTMCell_unsafe = unsafe_decorator(rnn_cell.BasicLSTMCell)
MultiRNNCell_unsafe = unsafe_decorator(rnn_cell.MultiRNNCell)
linear_unsafe = unsafe_decorator(linear)
multi_rnn_unsafe = unsafe_decorator(multi_rnn)
multi_bidirectional_rnn_unsafe = unsafe_decorator(multi_bidirectional_rnn)
