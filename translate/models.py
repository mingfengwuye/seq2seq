import tensorflow as tf
import functools
import math
from tensorflow.contrib.rnn import BasicLSTMCell, DropoutWrapper, RNNCell
from tensorflow.contrib.rnn import MultiRNNCell, LSTMStateTuple, GRUCell, LayerNormBasicLSTMCell
from translate.rnn import stack_bidirectional_dynamic_rnn
from translate import utils


def auto_reuse(fun):
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


get_variable = auto_reuse(tf.get_variable)
dense = auto_reuse(tf.layers.dense)


class CellWrapper(RNNCell):
    """
    Wrapper around LayerNormBasicLSTMCell, BasicLSTMCell and MultiRNNCell, to keep
    the state_is_tuple=False behavior (soon to be deprecated).
    """
    def __init__(self, cell):
        super(CellWrapper, self).__init__()
        self.cell = cell
        self.num_splits = len(cell.state_size) if isinstance(cell.state_size, tuple) else 1

    @property
    def state_size(self):
        return sum(self.cell.state_size)

    @property
    def output_size(self):
        return self.cell.output_size

    def __call__(self, inputs, state, scope=None):
        state = tf.split(value=state, num_or_size_splits=self.num_splits, axis=1)
        new_h, new_state = self.cell(inputs, state, scope=scope)
        return new_h, tf.concat(new_state, 1)


class ConditionalCellWrapper(RNNCell):
    def __init__(self, cell1, cell2):
        super(ConditionalCellWrapper, self).__init__()


def multi_encoder(encoder_inputs, encoders, encoder_input_length, other_inputs=None, dropout=None,
                  **kwargs):
    """
    Build multiple encoders according to the configuration in `encoders`, reading from `encoder_inputs`.
    The result is a list of the outputs produced by those encoders (for each time-step), and their final state.

    :param encoder_inputs: list of tensors of shape (batch_size, input_length), one tensor for each encoder.
    :param encoders: list of encoder configurations
    :param encoder_input_length: list of tensors of shape (batch_size,) (one tensor for each encoder)
    :param dropout: scalar tensor or None, specifying the keep probability (1 - dropout)
    :return:
      encoder outputs: a list of tensors of shape (batch_size, input_length, encoder_cell_size), hidden states of the
        encoders.
      encoder state: concatenation of the final states of all encoders, tensor of shape (batch_size, sum_of_state_sizes)
      new_encoder_input_length: list of tensors of shape (batch_size,) with the true length of the encoder outputs.
        May be different than `encoder_input_length` because of maxout strides, and time pooling.
    """
    encoder_states = []
    encoder_outputs = []

    # create embeddings in the global scope (allows sharing between encoder and decoder)
    embedding_variables = []
    for encoder in encoders:
        if encoder.binary:
            embedding_variables.append(None)
            continue
        # inputs are token ids, which need to be mapped to vectors (embeddings)
        embedding_shape = [encoder.vocab_size, encoder.embedding_size]
        # initializer = tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3))
        initializer = None

        device = '/cpu:0' if encoder.embeddings_on_cpu else None
        with tf.device(device):  # embeddings can take a very large amount of memory, so
            # storing them in GPU memory can be impractical
            embedding = get_variable('embedding_{}'.format(encoder.name), shape=embedding_shape,
                                     initializer=initializer)
        embedding_variables.append(embedding)

    new_encoder_input_length = []

    for i, encoder in enumerate(encoders):
        with tf.variable_scope('encoder_{}'.format(encoder.name)):
            encoder_inputs_ = encoder_inputs[i]
            encoder_input_length_ = encoder_input_length[i]

            def get_cell(reuse=False):
                if encoder.use_lstm:
                    cell = CellWrapper(BasicLSTMCell(encoder.cell_size, reuse=reuse))
                else:
                    cell = GRUCell(encoder.cell_size, reuse=reuse)

                if dropout is not None:
                    cell = DropoutWrapper(cell, input_keep_prob=dropout)

                return cell

            embedding = embedding_variables[i]

            batch_size = tf.shape(encoder_inputs_)[0]
            time_steps = tf.shape(encoder_inputs_)[1]

            if embedding is not None:
                flat_inputs = tf.reshape(encoder_inputs_, [tf.multiply(batch_size, time_steps)])
                flat_inputs = tf.nn.embedding_lookup(embedding, flat_inputs)
                encoder_inputs_ = tf.reshape(flat_inputs,
                                             tf.stack([batch_size, time_steps, flat_inputs.get_shape()[1].value]))

            if other_inputs is not None:
                encoder_inputs_ = tf.concat([encoder_inputs_, other_inputs], axis=2)

            if encoder.input_layers:
                for j, layer_size in enumerate(encoder.input_layers):
                    encoder_inputs_ = dense(encoder_inputs_, layer_size, activation=tf.tanh, use_bias=True,
                                            name='layer_{}'.format(j))
                    if dropout is not None:
                        encoder_inputs_ = tf.nn.dropout(encoder_inputs_, dropout)

            if encoder.convolutions:
                if encoder.binary:
                    raise NotImplementedError

                pad = tf.nn.embedding_lookup(embedding, utils.BOS_ID)
                pad = tf.expand_dims(tf.expand_dims(pad, axis=0), axis=1)
                pad = tf.tile(pad, [batch_size, 1, 1])

                # Fully Character-Level NMT without Explicit Segmentation, Lee et al. 2016
                inputs = []

                for w, filter_size in enumerate(encoder.convolutions, 1):
                    filter_ = get_variable('filter_{}'.format(w), [w, encoder.embedding_size, filter_size])

                    if w > 1:
                        right = (w - 1) // 2
                        left = (w - 1) - right
                        pad_right = tf.tile(pad, [1, right, 1])
                        pad_left = tf.tile(pad, [1, left, 1])
                        inputs_ = tf.concat([pad_left, encoder_inputs_, pad_right], axis=1)
                    else:
                        inputs_ = encoder_inputs_

                    inputs_ = tf.nn.convolution(inputs_, filter=filter_, padding='VALID')
                    inputs.append(inputs_)

                encoder_inputs_ = tf.concat(inputs, axis=2)
                # if encoder.convolution_activation.lower() == 'relu':
                encoder_inputs_ = tf.nn.relu(encoder_inputs_)

            if encoder.maxout_stride:
                if encoder.binary:
                    raise NotImplementedError

                stride = encoder.maxout_stride
                k = tf.to_int32(tf.ceil(time_steps / stride) * stride) - time_steps   # TODO: simpler
                pad = tf.zeros([batch_size, k, tf.shape(encoder_inputs_)[2]])
                encoder_inputs_ = tf.concat([encoder_inputs_, pad], axis=1)
                encoder_inputs_ = tf.nn.pool(encoder_inputs_, window_shape=[stride], pooling_type='MAX',
                                             padding='VALID', strides=[stride])
                encoder_input_length_ = tf.to_int32(tf.ceil(encoder_input_length_ / stride))

            # Contrary to Theano's RNN implementation, states after the sequence length are zero
            # (while Theano repeats last state)
            parameters = dict(
                inputs=encoder_inputs_, sequence_length=encoder_input_length_,
                dtype=tf.float32, parallel_iterations=encoder.parallel_iterations,
                time_pooling=encoder.time_pooling, pooling_avg=encoder.pooling_avg
            )

            state_size = get_cell().state_size
            if isinstance(state_size, LSTMStateTuple):
                state_size = state_size.c + state_size.h

            def get_initial_state(name='initial_state'):
                if encoder.trainable_initial_states:
                    initial_state = get_variable(name, initializer=tf.zeros(state_size))
                    return tf.tile(tf.expand_dims(initial_state, axis=0), [batch_size, 1])
                else:
                    return None

            if encoder.bidir:
                rnn = lambda reuse: stack_bidirectional_dynamic_rnn(
                    cells_fw=[get_cell(reuse=reuse) for _ in range(encoder.layers)],
                    cells_bw=[get_cell(reuse=reuse) for _ in range(encoder.layers)],
                    initial_states_fw=[get_initial_state('initial_state_fw')] * encoder.layers,
                    initial_states_bw=[get_initial_state('initial_state_bw')] * encoder.layers,
                    **parameters)

                try:
                    encoder_outputs_, _, backward_states = rnn(reuse=False)
                except ValueError:   # Multi-task scenario where we're reusing the same RNN parameters
                    encoder_outputs_, _, backward_states = rnn(reuse=True)

                if encoder.concat_last_states:
                    encoder_state_ = tf.concat(backward_states, axis=1)  # concats last states of all layers
                else:  # uses output, not state (which are different for LSTMs), and only the last layer's
                    encoder_state_ = encoder_outputs_[:, 0, encoder.cell_size:]  # first backward output

                if encoder.bidir_projection:
                    encoder_outputs_ = dense(encoder_outputs_, encoder.cell_size, use_bias=False,
                                             name='bidir_projection')

            else:
                if encoder.concat_last_states:
                    raise NotImplementedError
                if encoder.time_pooling:
                    raise NotImplementedError

                if encoder.layers > 1:
                    cell = MultiRNNCell([get_cell() for _ in range(encoder.layers)])
                    initial_state = (get_initial_state(),) * encoder.layers
                else:
                    cell = get_cell()
                    initial_state = get_initial_state()

                encoder_outputs_, _ = auto_reuse(tf.nn.dynamic_rnn)(cell=cell, initial_state=initial_state,
                                                                    **parameters)
                encoder_state_ = encoder_outputs_[:, -1, :]

            encoder_outputs.append(encoder_outputs_)
            encoder_states.append(encoder_state_)

            if encoder.time_pooling:
                for stride in encoder.time_pooling[:encoder.layers - 1]:
                    encoder_input_length_ = (encoder_input_length_ + stride - 1) // stride  # rounding up

            new_encoder_input_length.append(encoder_input_length_)

    encoder_state = tf.concat(encoder_states, 1)
    return encoder_outputs, encoder_state, new_encoder_input_length


def compute_energy(hidden, state, attn_size, **kwargs):
    y = dense(state, attn_size, use_bias=True, name='W_a')
    y = tf.expand_dims(y, axis=1)
    f = dense(hidden, attn_size, use_bias=False, name='U_a')

    v = get_variable('v_a', [attn_size])
    s = f + y

    return tf.reduce_sum(v * tf.tanh(s), axis=2)


def compute_energy_with_filter(hidden, state, prev_weights, attention_filters, attention_filter_length,
                               **kwargs):
    hidden = tf.expand_dims(hidden, 2)

    batch_size = tf.shape(hidden)[0]
    time_steps = tf.shape(hidden)[1]
    attn_size = hidden.get_shape()[3].value

    filter_shape = [attention_filter_length * 2 + 1, 1, 1, attention_filters]
    filter_ = get_variable('filter', filter_shape)
    u = get_variable('U', [attention_filters, attn_size])
    prev_weights = tf.reshape(prev_weights, tf.stack([batch_size, time_steps, 1, 1]))
    conv = tf.nn.conv2d(prev_weights, filter_, [1, 1, 1, 1], 'SAME')
    shape = tf.stack([tf.multiply(batch_size, time_steps), attention_filters])
    conv = tf.reshape(conv, shape)
    z = tf.matmul(conv, u)
    z = tf.reshape(z, tf.stack([batch_size, time_steps, 1, attn_size]))

    y = dense(state, attn_size, use_bias=True, name='y')
    y = tf.reshape(y, [-1, 1, 1, attn_size])

    k = get_variable('W', [attn_size, attn_size])
    # dot product between tensors requires reshaping
    hidden = tf.reshape(hidden, tf.stack([tf.multiply(batch_size, time_steps), attn_size]))
    f = tf.matmul(hidden, k)
    f = tf.reshape(f, tf.stack([batch_size, time_steps, 1, attn_size]))

    v = get_variable('V', [attn_size])
    s = f + y + z
    return tf.reduce_sum(v * tf.tanh(s), [2, 3])


def global_attention(state, hidden_states, encoder, encoder_input_length, scope=None, context=None, **kwargs):
    with tf.variable_scope(scope or 'attention_{}'.format(encoder.name)):
        if context is not None and encoder.use_context:
            state = tf.concat([state, context], axis=1)

        if encoder.attention_filters:
            e = compute_energy_with_filter(hidden_states, state, attn_size=encoder.attn_size,
                                           attention_filters=encoder.attention_filters,
                                           attention_filter_length=encoder.attention_filter_length, **kwargs)
        else:
            e = compute_energy(hidden_states, state, attn_size=encoder.attn_size, **kwargs)

        e -= tf.reduce_max(e, axis=1, keep_dims=True)
        mask = tf.sequence_mask(encoder_input_length, maxlen=tf.shape(hidden_states)[1], dtype=tf.float32)

        T = encoder.attention_temperature or 1.0
        exp = tf.exp(T * e) * mask
        weights = exp / tf.reduce_sum(exp, axis=-1, keep_dims=True)
        weighted_average = tf.reduce_sum(tf.expand_dims(weights, 2) * hidden_states, axis=1)

        return weighted_average, weights


def no_attention(hidden_states, *args, **kwargs):
    batch_size = tf.shape(hidden_states)[0]
    weighted_average = tf.zeros(shape=tf.stack([batch_size, 0]))
    weights = tf.zeros(shape=tf.shape(hidden_states)[:2])
    return weighted_average, weights


def average_attention(hidden_states, encoder_input_length, *args, **kwargs):
    # attention with fixed weights (average of all hidden states)
    lengths = tf.to_float(tf.expand_dims(encoder_input_length, axis=1))
    mask = tf.sequence_mask(encoder_input_length, maxlen=tf.shape(hidden_states)[1])
    weights = tf.to_float(mask) / lengths
    weighted_average = tf.reduce_sum(hidden_states * tf.expand_dims(weights, axis=2), axis=1)
    return weighted_average, weights


def last_state_attention(hidden_states, encoder_input_length, *args, **kwargs):
    weights = tf.one_hot(encoder_input_length - 1, tf.shape(hidden_states)[1])
    weights = tf.to_float(weights)

    weighted_average = tf.reduce_sum(hidden_states * tf.expand_dims(weights, axis=2), axis=1)
    return weighted_average, weights


def local_attention(state, hidden_states, encoder, encoder_input_length, pos=None, scope=None,
                    context=None, **kwargs):
    batch_size = tf.shape(state)[0]
    attn_length = tf.shape(hidden_states)[1]

    if context is not None and encoder.use_context:
        state = tf.concat([state, context], axis=1)

    state_size = state.get_shape()[1].value

    with tf.variable_scope(scope or 'attention_{}'.format(encoder.name)):
        encoder_input_length = tf.to_float(tf.expand_dims(encoder_input_length, axis=1))

        if pos is not None:
            pos = tf.reshape(pos, [-1, 1])
            pos = tf.minimum(pos, encoder_input_length - 1)

        if pos is not None and encoder.attention_window_size > 0:
            # `pred_edits` scenario, where we know the aligned pos
            # when the windows size is non-zero, we concatenate consecutive encoder states
            # and map it to the right attention vector size.
            weights = tf.to_float(tf.one_hot(tf.to_int32(tf.squeeze(pos, axis=1)), depth=attn_length))

            weighted_average = []
            for offset in range(-encoder.attention_window_size, encoder.attention_window_size + 1):
                pos_ = pos + offset
                pos_ = tf.minimum(pos_, encoder_input_length - 1)
                pos_ = tf.maximum(pos_, 0)  # TODO: when pos is < 0, use <S> or </S>
                weights_ = tf.to_float(tf.one_hot(tf.to_int32(tf.squeeze(pos_, axis=1)), depth=attn_length))
                weighted_average_ = tf.reduce_sum(tf.expand_dims(weights_, axis=2) * hidden_states, axis=1)
                weighted_average.append(weighted_average_)

            weighted_average = tf.concat(weighted_average, axis=1)
            weighted_average = dense(weighted_average, encoder.attn_size)
        elif pos is not None:
            weights = tf.to_float(tf.one_hot(tf.to_int32(tf.squeeze(pos, axis=1)), depth=attn_length))
            weighted_average = tf.reduce_sum(tf.expand_dims(weights, axis=2) * hidden_states, axis=1)
        else:
            # Local attention of Luong et al. (http://arxiv.org/abs/1508.04025)
            wp = get_variable('Wp', [state_size, state_size])
            vp = get_variable('vp', [state_size, 1])

            pos = tf.nn.sigmoid(tf.matmul(tf.nn.tanh(tf.matmul(state, wp)), vp))
            pos = tf.floor(encoder_input_length * pos)
            pos = tf.reshape(pos, [-1, 1])
            pos = tf.minimum(pos, encoder_input_length - 1)

            idx = tf.tile(tf.to_float(tf.range(attn_length)), tf.stack([batch_size]))
            idx = tf.reshape(idx, [-1, attn_length])

            low = pos - encoder.attention_window_size
            high = pos + encoder.attention_window_size

            mlow = tf.to_float(idx < low)
            mhigh = tf.to_float(idx > high)
            m = mlow + mhigh
            m += tf.to_float(idx >= encoder_input_length)

            mask = tf.to_float(tf.equal(m, 0.0))

            e = compute_energy(hidden_states, state, attn_size=encoder.attn_size, **kwargs)

            weights = softmax(e, mask=mask)

            sigma = encoder.attention_window_size / 2
            numerator = -tf.pow((idx - pos), tf.convert_to_tensor(2, dtype=tf.float32))
            div = tf.truediv(numerator, 2 * sigma ** 2)
            weights *= tf.exp(div)  # result of the truncated normal distribution
            # normalize to keep a probability distribution
            # weights /= (tf.reduce_sum(weights, axis=1, keep_dims=True) + 10e-12)

            weighted_average = tf.reduce_sum(tf.expand_dims(weights, axis=2) * hidden_states, axis=1)

        return weighted_average, weights


def attention(encoder, **kwargs):
    attention_functions = {
        'global': global_attention,
        'local': local_attention,
        'none': no_attention,
        'average': average_attention,
        'last_state': last_state_attention
    }

    attention_function = attention_functions.get(encoder.attention_type, global_attention)

    return attention_function(encoder=encoder, **kwargs)


def multi_attention(state, hidden_states, encoders, encoder_input_length, pos=None, aggregation_method='sum',
                    prev_weights=None, **kwargs):
    attns = []
    weights = []

    context_vector = None
    for i, (hidden, encoder, input_length) in enumerate(zip(hidden_states, encoders, encoder_input_length)):
        pos_ = pos[i] if pos is not None else None
        prev_weights_ = prev_weights[i] if prev_weights is not None else None
        context_vector, weights_ = attention(state=state, hidden_states=hidden, encoder=encoder,
                                             encoder_input_length=input_length, pos=pos_, context=context_vector,
                                             prev_weights=prev_weights_, **kwargs)
        attns.append(context_vector)
        weights.append(weights_)

    if aggregation_method == 'sum':
        context_vector = tf.reduce_sum(tf.stack(attns, axis=2), axis=2)
    else:
        context_vector = tf.concat(attns, axis=1)

    return context_vector, weights


def attention_decoder(decoder_inputs, initial_state, attention_states, encoders, decoder, encoder_input_length,
                      dropout=None, feed_previous=0.0, align_encoder_id=0, feed_argmax=True, **kwargs):
    """
    :param decoder_inputs: int32 tensor of shape (batch_size, output_length)
    :param initial_state: initial state of the decoder (usually the final state of the encoder),
      as a float32 tensor of shape (batch_size, initial_state_size). This state is mapped to the
      correct state size for the decoder.
    :param attention_states: list of tensors of shape (batch_size, input_length, encoder_cell_size),
      the hidden states of the encoder(s) (one tensor for each encoder).
    :param encoders: configuration of the encoders
    :param decoder: configuration of the decoder
    :param encoder_input_length: list of int32 tensors of shape (batch_size,), tells for each encoder,
     the true length of each sequence in the batch (sequences in the same batch are padded to all have the same
     length).
    :param dropout: scalar tensor or None, specifying the keep probability (1 - dropout)
    :param feed_previous: scalar tensor corresponding to the probability to use previous decoder output
      instead of the ground truth as input for the decoder (1 when decoding, between 0 and 1 when training)
    :param feed_argmax: boolean tensor, when True the greedy decoder outputs the word with the highest
    probability (argmax). When False, it samples a word from the probability distribution (softmax).
    :param align_encoder_id: outputs attention weights for this encoder. Also used when predicting edit operations
    (pred_edits), to specifify which encoder reads the sequence to post-edit (MT).

    :return:
      outputs of the decoder as a tensor of shape (batch_size, output_length, decoder_cell_size)
      attention weights as a tensor of shape (output_length, encoders, batch_size, input_length)
    """
    assert not decoder.output_maxout or decoder.cell_size % 2 == 0, 'cell size must be a multiple of 2'

    embedding_shape = [decoder.vocab_size, decoder.embedding_size]

    device = '/cpu:0' if decoder.embeddings_on_cpu else None
    with tf.device(device):
        embedding = get_variable('embedding_{}'.format(decoder.name), shape=embedding_shape)

    def embed(input_):
        return tf.nn.embedding_lookup(embedding, input_)

    def get_cell(reuse=False):
        cells = []

        for _ in range(decoder.layers):
            if decoder.use_lstm:
                cell = CellWrapper(BasicLSTMCell(decoder.cell_size, reuse=reuse))
            else:
                cell = GRUCell(decoder.cell_size, reuse=reuse)

            if dropout is not None:
                cell = DropoutWrapper(cell, input_keep_prob=dropout)

            cells.append(cell)

        if len(cells) == 1:
            return cells[0]
        else:
            return CellWrapper(MultiRNNCell(cells))

    def get_attention(state, input_, prev_weights=None, pos=None):
        prev_weights_ = [prev_weights if i == align_encoder_id else None for i in range(len(encoders))]
        pos_ = None
        if decoder.pred_edits:
            pos_ = [pos if i == align_encoder_id else None for i in range(len(encoders))]
        if decoder.attn_prev_word:
            state = tf.concat([state, input_], axis=1)

        parameters = dict(hidden_states=attention_states, encoder_input_length=encoder_input_length,
                          encoders=encoders, aggregation_method=decoder.aggregation_method)
        context, new_weights = multi_attention(state, pos=pos_, prev_weights=prev_weights_, **parameters)
        return context, new_weights[align_encoder_id]

    def get_next_state(state, input_, context, symbol=None):
        if decoder.input_attention:
            input_ = tf.concat([input_, context], axis=1)

        try:
            _, new_state = get_cell()(input_, state)
        except ValueError:  # auto_reuse doesn't work with LSTM cells
            _, new_state = get_cell(reuse=True)(input_, state)

        if decoder.skip_update and decoder.pred_edits and symbol is not None:
            is_del = tf.equal(symbol, utils.DEL_ID)
            new_state = tf.where(is_del, state, new_state)

        return new_state

    def get_next_pos(pos, symbol, max_pos=None):
        if not decoder.pred_edits:
            return pos

        is_keep = tf.equal(symbol, utils.KEEP_ID)
        is_del = tf.equal(symbol, utils.DEL_ID)
        is_not_ins = tf.logical_or(is_keep, is_del)
        pos += tf.to_float(is_not_ins)
        if max_pos is not None:
            pos = tf.minimum(pos, tf.to_float(max_pos))
        return pos

    input_shape = tf.shape(decoder_inputs)
    batch_size = input_shape[0]
    time_steps = input_shape[1]

    output_size = decoder.vocab_size

    state_size = get_cell().state_size
    cell_output_size = get_cell().output_size

    time = tf.constant(0, dtype=tf.int32, name='time')

    outputs = tf.TensorArray(dtype=tf.float32, size=time_steps)
    samples = tf.TensorArray(dtype=tf.int64, size=time_steps)
    inputs = tf.TensorArray(dtype=tf.int64, size=time_steps, clear_after_read=False).unstack(
                            tf.to_int64(tf.transpose(decoder_inputs, perm=(1, 0))))
    states = tf.TensorArray(dtype=tf.float32, size=time_steps)
    weights = tf.TensorArray(dtype=tf.float32, size=time_steps)
    attns = tf.TensorArray(dtype=tf.float32, size=time_steps)

    initial_symbol = inputs.read(0)  # first symbol is BOS
    initial_input = embed(initial_symbol)
    initial_pos = tf.zeros([batch_size], tf.float32)
    initial_weights = tf.zeros(tf.shape(attention_states[align_encoder_id])[:2])
    #initial_context, _ = attention_(initial_input, pos=initial_pos, prev_weights=initial_weights)

    if dropout is not None:
        initial_state = tf.nn.dropout(initial_state, dropout)

    with tf.variable_scope('decoder_{}'.format(decoder.name)):
        initial_state = dense(initial_state, state_size, use_bias=True, name='initial_state_projection',
                              activation=tf.nn.tanh)

    # `zero_state` means that the initial state is not the first state of our decoder (used to predict the
    # first output word), but the state before it.
    # Using the attention context as input to the decoder does not make sense in this case,
    # as we would use the same context vector for state zero and state one.
    if decoder.zero_state and not decoder.vanilla and not decoder.input_attention:
        initial_state = get_next_state(initial_state, initial_input, context=None, symbol=None)

    initial_data = tf.concat([initial_state, tf.expand_dims(initial_pos, axis=1), initial_weights], axis=1)
    initial_state, initial_pos, initial_weights = tf.split(initial_data, [state_size, 1, -1], axis=1)
    initial_state.set_shape([None, state_size])
    initial_pos = initial_pos[:, 0]

    def _time_step(time, input_, input_symbol, pos, state, outputs, states, weights, attns, prev_weights, samples):
        # FIXME: use LSTM state or LSTM output
        context, new_weights = get_attention(state, input_, pos=pos, prev_weights=prev_weights)

        attns = attns.write(time, context)
        weights = weights.write(time, new_weights)

        if decoder.vanilla:
            state = get_next_state(state, input_, context, input_symbol)

        states = states.write(time, state)

        rnn_output = state[:, -cell_output_size:]
        projection_input = [state] if decoder.use_lstm_state else [rnn_output]
        if decoder.use_previous_word:
            projection_input.append(input_)
        projection_input.append(context)

        output_ = tf.concat(projection_input, axis=1)

        if decoder.output_maxout:
            output_ = dense(output_, decoder.cell_size, use_bias=False, name='maxout')
            output_ = tf.nn.pool(tf.expand_dims(output_, axis=2), window_shape=[2], pooling_type='MAX',
                                padding='SAME', strides=[2])
            output_ = tf.squeeze(output_, axis=2)

        if decoder.output_extra_proj:
            # intermediate projection to embedding size (before projecting to vocabulary size)
            # this is useful to reduce the number of parameters, and
            # to use the output embeddings for output projection (tie_embeddings parameter)
            output_ = dense(output_, decoder.embedding_size, use_bias=False, name='softmax0')

        if decoder.output_extra_proj and decoder.tie_embeddings:
            bias = get_variable('softmax1/bias', shape=[decoder.vocab_size])
            output_ = tf.matmul(output_, tf.transpose(embedding)) + bias
        else:
            output_ = dense(output_, output_size, use_bias=True, name='softmax1')

        outputs = outputs.write(time, output_)

        argmax = lambda: tf.argmax(output_, 1)
        target = lambda: inputs.read(time + 1)
        softmax = lambda: tf.squeeze(tf.multinomial(tf.log(tf.nn.softmax(output_)), num_samples=1),
                                     axis=1)

        predicted_symbol = tf.case([
            (tf.logical_and(time < time_steps - 1, tf.random_uniform([]) >= feed_previous), target),
            (tf.logical_not(feed_argmax), softmax)],
            default=argmax)   # default case is useful for beam-search
        predicted_symbol.set_shape([None])
        predicted_symbol = tf.stop_gradient(predicted_symbol)
        samples = samples.write(time, predicted_symbol)

        input_ = embed(predicted_symbol)
        pos = get_next_pos(pos, predicted_symbol, encoder_input_length[align_encoder_id])

        if not decoder.vanilla:
            state = get_next_state(state, input_, context, predicted_symbol)

        return (time + 1, input_, predicted_symbol, pos, state, outputs, states, weights, attns, new_weights,
                samples)

    with tf.variable_scope('decoder_{}'.format(decoder.name)):
        _, _, _, new_pos, new_state, outputs, states, weights, attns, new_weights, samples = tf.while_loop(
            cond=lambda time, *_: time < time_steps,
            body=_time_step,
            loop_vars=(time, initial_input, initial_symbol, initial_pos, initial_state, outputs, weights, states,
                       attns, initial_weights, samples),
            parallel_iterations=decoder.parallel_iterations,
            swap_memory=decoder.swap_memory)

    outputs = outputs.stack()
    weights = weights.stack()  # batch_size, encoders, output time, input time
    states = states.stack()
    attns = attns.stack()
    samples = samples.stack()

    new_data = tf.concat([new_state, tf.expand_dims(new_pos, axis=1), new_weights], axis=1)
    beam_tensors = utils.AttrDict(data=initial_data, new_data=new_data)

    # put batch_size as first dimension
    outputs = tf.transpose(outputs, perm=(1, 0, 2))
    weights = tf.transpose(weights[1:], perm=(1, 0, 2))
    states = tf.transpose(states, perm=(1, 0, 2))
    attns = tf.transpose(attns, perm=(1, 0, 2))
    samples = tf.transpose(samples)

    return outputs, weights, states, attns, beam_tensors, samples


def encoder_decoder(encoders, decoders, dropout, encoder_inputs, targets, feed_previous, align_encoder_id=0,
                    encoder_input_length=None, feed_argmax=True, rewards=None, use_baseline=True, **kwargs):
    decoder = decoders[0]
    targets = targets[0]  # single decoder

    if encoder_input_length is None:
        encoder_input_length = []
        for encoder_inputs_ in encoder_inputs:
            weights = get_weights(encoder_inputs_, utils.EOS_ID, include_first_eos=True)
            encoder_input_length.append(tf.to_int32(tf.reduce_sum(weights, axis=1)))

    parameters = dict(encoders=encoders, decoder=decoder, dropout=dropout, encoder_inputs=encoder_inputs,
                      feed_argmax=feed_argmax)

    target_weights = get_weights(targets[:, 1:], utils.EOS_ID, include_first_eos=True)

    attention_states, encoder_state, encoder_input_length = multi_encoder(
        encoder_input_length=encoder_input_length, **parameters)

    outputs, attention_weights, _, _, beam_tensors, samples = attention_decoder(
        attention_states=attention_states, initial_state=encoder_state, feed_previous=feed_previous,
        decoder_inputs=targets[:, :-1], align_encoder_id=align_encoder_id, encoder_input_length=encoder_input_length,
        **parameters
    )

    if use_baseline:
        baseline_rewards = reinforce_baseline(outputs, rewards)   # FIXME: use outputs or decoder states/outputs?
        baseline_weights = get_weights(samples, utils.EOS_ID, include_first_eos=False)
        baseline_loss_ = baseline_loss(rewards=baseline_rewards, weights=baseline_weights)
    else:
        baseline_rewards = rewards
        baseline_loss_ = tf.constant(0.0)

    reinforce_weights = get_weights(samples, utils.EOS_ID, include_first_eos=True)
    reinforce_loss = sequence_loss(logits=outputs, targets=samples, weights=reinforce_weights,
                                   rewards=baseline_rewards)

    xent_loss = sequence_loss(logits=outputs, targets=targets[:, 1:], weights=target_weights)
    losses = [xent_loss, reinforce_loss, baseline_loss_]

    return losses, [outputs], encoder_state, attention_states, attention_weights, beam_tensors, samples


def chained_encoder_decoder(encoders, decoders, dropout, encoder_inputs, targets, feed_previous,
                            chaining_strategy=None, more_dropout=False, align_encoder_id=0,
                            chaining_non_linearity=False, chaining_loss_ratio=1.0, chaining_stop_gradient=False,
                            **kwargs):
    decoder = decoders[0]
    targets = targets[0]  # single decoder

    assert len(encoders) == 2

    encoder_input_length = []
    input_weights = []
    for encoder_inputs_ in encoder_inputs:
        weights = get_weights(encoder_inputs_, utils.EOS_ID, include_first_eos=True)
        input_weights.append(weights)
        encoder_input_length.append(tf.to_int32(tf.reduce_sum(weights, axis=1)))

    target_weights = get_weights(targets[:, 1:], utils.EOS_ID, include_first_eos=True)

    parameters = dict(encoders=encoders[1:], decoder=encoders[0], dropout=dropout, more_dropout=more_dropout)

    attention_states, encoder_state, encoder_input_length[1:] = multi_encoder(
        encoder_inputs[1:], encoder_input_length=encoder_input_length[1:], **parameters)

    decoder_inputs = encoder_inputs[0][:, :-1]
    batch_size = tf.shape(decoder_inputs)[0]

    pad = tf.ones(shape=tf.stack([batch_size, 1]), dtype=tf.int32) * utils.BOS_ID
    decoder_inputs = tf.concat([pad, decoder_inputs], axis=1)

    outputs, _, states, attns, _, samples = attention_decoder(
        attention_states=attention_states, initial_state=encoder_state, decoder_inputs=decoder_inputs,
        encoder_input_length=encoder_input_length[1:], **parameters
    )

    chaining_loss = sequence_loss(logits=outputs, targets=encoder_inputs[0], weights=input_weights[0])

    if decoder.use_lstm:
        size = states.get_shape()[2].value
        decoder_outputs = states[:, :, size // 2:]
    else:
        decoder_outputs = states

    if chaining_strategy == 'share_states':
        other_inputs = states
    elif chaining_strategy == 'share_outputs':
        other_inputs = decoder_outputs
    else:
        other_inputs = None

    if other_inputs is not None and chaining_stop_gradient:
        other_inputs = tf.stop_gradient(other_inputs)

    parameters = dict(encoders=encoders[:1], decoder=decoder, dropout=dropout, encoder_inputs=encoder_inputs[:1],
                      other_inputs=other_inputs)

    attention_states, encoder_state, encoder_input_length[:1] = multi_encoder(
        encoder_input_length=encoder_input_length[:1], **parameters)

    if dropout is not None and more_dropout:
        attns = tf.nn.dropout(attns, keep_prob=dropout)
        states = tf.nn.dropout(states, keep_prob=dropout)
        decoder_outputs = tf.nn.dropout(decoder_outputs, keep_prob=dropout)

    if chaining_stop_gradient:
        attns = tf.stop_gradient(attns)
        states = tf.stop_gradient(states)
        decoder_outputs = tf.stop_gradient(decoder_outputs)

    if chaining_strategy == 'concat_attns':
        attention_states[0] = tf.concat([attention_states[0], attns], axis=2)
    elif chaining_strategy == 'concat_states':
        attention_states[0] = tf.concat([attention_states[0], states], axis=2)
    elif chaining_strategy == 'sum_attns':
        attention_states[0] += attns
    elif chaining_strategy in ('map_attns', 'map_states', 'map_outputs'):
        if chaining_strategy == 'map_attns':
            x = attns
        elif chaining_strategy == 'map_outputs':
            x = decoder_outputs
        else:
            x = states

        shape = [x.get_shape()[-1], attention_states[0].get_shape()[-1]]

        w = tf.get_variable("map_attns/matrix", shape=shape)
        b = tf.get_variable("map_attns/bias", shape=shape[-1:])

        x = tf.einsum('ijk,kl->ijl', x, w) + b
        if chaining_non_linearity:
            x = tf.nn.tanh(x)
            if dropout is not None and more_dropout:
                x = tf.nn.dropout(x, keep_prob=dropout)

        attention_states[0] += x

    outputs, attention_weights, _, _, beam_tensors, samples = attention_decoder(
        attention_states=attention_states, initial_state=encoder_state,
        feed_previous=feed_previous, decoder_inputs=targets[:,:-1],
        align_encoder_id=align_encoder_id, encoder_input_length=encoder_input_length[:1],
        **parameters
    )

    xent_loss = sequence_loss(logits=outputs, targets=targets[:, 1:],
                              weights=target_weights)

    if chaining_loss is not None and chaining_loss_ratio:
        xent_loss += chaining_loss_ratio * chaining_loss

    losses = [xent_loss, None, None]

    return losses, [outputs], encoder_state, attention_states, attention_weights, beam_tensors, samples


def softmax(logits, dim=-1, mask=None):
    e = tf.exp(logits)
    if mask is not None:
        e *= mask

    return e / tf.clip_by_value(tf.reduce_sum(e, axis=dim, keep_dims=True), 10e-37, 10e+37)


def sequence_loss(logits, targets, weights, average_across_timesteps=False, average_across_batch=True, rewards=None):
    batch_size = tf.shape(targets)[0]
    time_steps = tf.shape(targets)[1]

    logits_ = tf.reshape(logits, tf.stack([time_steps * batch_size, logits.get_shape()[2].value]))
    targets_ = tf.reshape(targets, tf.stack([time_steps * batch_size]))

    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_, labels=targets_)
    crossent = tf.reshape(crossent, tf.stack([batch_size, time_steps]))

    if rewards is not None:
        crossent *= tf.stop_gradient(rewards)

    log_perp = tf.reduce_sum(crossent * weights, axis=1)

    if average_across_timesteps:
        total_size = tf.reduce_sum(weights, axis=0)
        total_size += 1e-12  # just to avoid division by 0 for all-0 weights
        log_perp /= total_size

    cost = tf.reduce_sum(log_perp)

    if average_across_batch:
        return cost / tf.to_float(batch_size)
    else:
        return cost


def get_weights(sequence, eos_id, include_first_eos=True):
    weights = (1.0 - tf.minimum(
        tf.cumsum(tf.to_float(tf.equal(sequence, eos_id)), axis=1), 1.0))

    if include_first_eos:
        weights = weights[:,:-1]
        shape = [tf.shape(weights)[0], 1]
        weights = tf.concat([tf.ones(tf.stack(shape)), weights], axis=1)

    return tf.stop_gradient(weights)


def reinforce_baseline(decoder_states, reward):
    """
    Center the reward by computing a baseline reward over decoder states.

    :param decoder_states: internal states of the decoder, tensor of shape (batch_size, time_steps, state_size)
    :param reward: reward for each time step, tensor of shape (batch_size, time_steps)
    :return: reward - computed baseline, tensor of shape (batch_size, time_steps)
    """
    # batch_size = tf.shape(decoder_states)[0]
    # time_steps = tf.shape(decoder_states)[1]
    # state_size = decoder_states.get_shape()[2]
    # states = tf.reshape(decoder_states, shape=tf.stack([batch_size * time_steps, state_size]))

    baseline = dense(tf.stop_gradient(decoder_states), units=1, activation=None, name='reward_baseline',
                     kernel_initializer=tf.constant_initializer(0.01))
    baseline = tf.squeeze(baseline, axis=2)

    # baseline = tf.reshape(baseline, shape=tf.stack([batch_size, time_steps]))
    return reward - baseline


def baseline_loss(rewards, weights, average_across_timesteps=False, average_across_batch=True):
    """
    :param reward: tensor of shape (batch_size, time_steps)
    :param weights: tensor of shape (batch_size, time_steps)
    """
    batch_size = tf.shape(rewards)[0]

    cost = rewards ** 2
    cost = tf.reduce_sum(cost * weights, axis=1)

    if average_across_timesteps:
        total_size = tf.reduce_sum(weights, axis=1)
        total_size += 1e-12  # just to avoid division by 0 for all-0 weights
        cost /= total_size

    cost = tf.reduce_sum(cost)

    if average_across_batch:
        cost /= tf.to_float(batch_size)

    return cost

