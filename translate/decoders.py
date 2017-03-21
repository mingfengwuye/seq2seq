import tensorflow as tf
import functools
import math
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn import BasicLSTMCell, DropoutWrapper
from tensorflow.contrib.layers import fully_connected
from translate.rnn import get_variable_unsafe, linear_unsafe, multi_rnn_unsafe, orthogonal_initializer
from translate.rnn import multi_bidirectional_rnn_unsafe, unsafe_decorator, MultiRNNCell, GRUCell
from translate import utils
from collections import namedtuple


def multi_encoder(encoder_inputs, encoders, encoder_input_length, dropout=None, **kwargs):
    """
    Build multiple encoders according to the configuration in `encoders`, reading from `encoder_inputs`.
    The result is a list of the outputs produced by those encoders (for each time-step), and their final state.

    :param encoder_inputs: list of tensors of shape (batch_size, input_length) (one tensor for each encoder)
    :param encoders: list of encoder configurations
    :param encoder_input_length: list of tensors of shape (batch_size) (one tensor for each encoder)
    :param dropout: scalar tensor or None, specifying the keep probability (1 - dropout)
    :return:
      encoder outputs: a list of tensors of shape (batch_size, input_length, encoder_cell_size)
      encoder state: concatenation of the final states of all encoders, tensor of shape (batch_size, sum_of_state_sizes)
    """
    assert len(encoder_inputs) == len(encoders)
    encoder_states = []
    encoder_outputs = []

    # create embeddings in the global scope (allows sharing between encoder and decoder)
    embedding_variables = []
    for encoder in encoders:
        # inputs are token ids, which need to be mapped to vectors (embeddings)
        if not encoder.binary:
            if encoder.get('embedding') is not None:
                initializer = encoder.embedding
                embedding_shape = None
            else:
                # initializer = tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3))
                initializer = None
                embedding_shape = [encoder.vocab_size, encoder.embedding_size]

            with tf.device('/cpu:0'):
                embedding = get_variable_unsafe('embedding_{}'.format(encoder.name), shape=embedding_shape,
                                                initializer=initializer)
            embedding_variables.append(embedding)
        else:  # do nothing: inputs are already vectors
            embedding_variables.append(None)

    for i, encoder in enumerate(encoders):
        with tf.variable_scope('encoder_{}'.format(encoder.name)):
            encoder_inputs_ = encoder_inputs[i]
            encoder_input_length_ = encoder_input_length[i]

            # TODO: use state_is_tuple=True
            if encoder.use_lstm:
                cell = BasicLSTMCell(encoder.cell_size, state_is_tuple=False)
            else:
                cell = GRUCell(encoder.cell_size, initializer=orthogonal_initializer())

            if dropout is not None:
                cell = DropoutWrapper(cell, input_keep_prob=dropout)

            embedding = embedding_variables[i]

            if embedding is not None or encoder.input_layers:
                batch_size = tf.shape(encoder_inputs_)[0]  # TODO: fix this time major stuff
                time_steps = tf.shape(encoder_inputs_)[1]

                if embedding is None:
                    size = encoder_inputs_.get_shape()[2].value
                    flat_inputs = tf.reshape(encoder_inputs_, [tf.multiply(batch_size, time_steps), size])
                else:
                    flat_inputs = tf.reshape(encoder_inputs_, [tf.multiply(batch_size, time_steps)])
                    flat_inputs = tf.nn.embedding_lookup(embedding, flat_inputs)

                if encoder.input_layers:
                    for j, size in enumerate(encoder.input_layers):
                        name = 'input_layer_{}'.format(j)
                        flat_inputs = tf.nn.tanh(linear_unsafe(flat_inputs, size, bias=True, scope=name))
                        if dropout is not None:
                            flat_inputs = tf.nn.dropout(flat_inputs, dropout)

                encoder_inputs_ = tf.reshape(flat_inputs,
                                             tf.stack([batch_size, time_steps, flat_inputs.get_shape()[1].value]))

            # Contrary to Theano's RNN implementation, states after the sequence length are zero
            # (while Theano repeats last state)
            sequence_length = encoder_input_length_   # TODO
            parameters = dict(
                inputs=encoder_inputs_, sequence_length=sequence_length, time_pooling=encoder.time_pooling,
                pooling_avg=encoder.pooling_avg, dtype=tf.float32, swap_memory=encoder.swap_memory,
                parallel_iterations=encoder.parallel_iterations, residual_connections=encoder.residual_connections,
                trainable_initial_state=True
            )

            if encoder.bidir:
                encoder_outputs_, _, _ = multi_bidirectional_rnn_unsafe(
                    cells=[(cell, cell)] * encoder.layers, **parameters)
                # Like Bahdanau et al., we use the first annotation h_1 of the backward encoder
                encoder_state_ = encoder_outputs_[:, 0, encoder.cell_size:]
                # TODO: if multiple layers, combine last states with a Maxout layer
            else:
                encoder_outputs_, encoder_state_ = multi_rnn_unsafe(
                    cells=[cell] * encoder.layers, **parameters)
                encoder_state_ = encoder_outputs_[:, -1, :]

            encoder_outputs.append(encoder_outputs_)
            encoder_states.append(encoder_state_)

    encoder_state = tf.concat(encoder_states, 1)
    return tf.stack(encoder_outputs), encoder_state


def compute_energy(hidden, state, attn_size, **kwargs):
    input_size = hidden.get_shape()[2].value

    y = linear_unsafe(state, attn_size, True, scope='W_a')
    y = tf.expand_dims(y, axis=1)

    k = get_variable_unsafe('U_a', [input_size, attn_size])
    f = tf.einsum('ijk,kl->ijl', hidden, k)

    v = get_variable_unsafe('v_a', [attn_size])
    s = f + y

    return tf.reduce_sum(v * tf.tanh(s), [2])


def compute_energy_with_filter(hidden, state, prev_weights, attention_filters, attention_filter_length, **kwargs):
    batch_size = tf.shape(hidden)[0]
    time_steps = tf.shape(hidden)[1]
    attn_size = hidden.get_shape()[2].value  # FIXME attn_size parameter

    filter_shape = [attention_filter_length * 2 + 1, 1, 1, attention_filters]
    filter_ = get_variable_unsafe('filter', filter_shape)
    u = get_variable_unsafe('U', [attention_filters, attn_size])
    prev_weights = tf.reshape(prev_weights, tf.stack([batch_size, time_steps, 1, 1]))
    conv = tf.nn.conv2d(prev_weights, filter_, [1, 1, 1, 1], 'SAME')

    z = tf.einsum('ijk,kl->ijl', tf.squeeze(conv, axis=2), u)

    y = linear_unsafe(state, attn_size, True)
    y = tf.expand_dims(y, axis=1)

    k = get_variable_unsafe('W', [attn_size, attn_size])
    f = tf.einsum('ijk,kl->ijl', hidden, k)

    v = get_variable_unsafe('V', [attn_size])
    s = f + y + z
    return tf.reduce_sum(v * tf.tanh(s), axis=2)


def compute_energy_edits(hidden, state, attn_size, edit_window_size=3, pos=None, **kwargs):
    input_size = hidden.get_shape()[2].value
    batch_size = tf.shape(hidden)[0]
    time_steps = tf.shape(hidden)[1]

    y = linear_unsafe(state, attn_size, bias=True, scope='W_a')
    y = tf.expand_dims(y, axis=1)

    k = get_variable_unsafe('U_a', [input_size, attn_size])
    f = tf.einsum('ijk,kl->ijl', hidden, k)

    i = tf.reshape(tf.tile(tf.range(time_steps), [batch_size]), tf.stack([batch_size, time_steps]))
    i -= tf.cast(tf.expand_dims(pos, axis=1), tf.int32)

    i = tf.maximum(tf.minimum(i, edit_window_size), -edit_window_size) + edit_window_size

    pos_embedding = get_variable_unsafe('pos_embedding', [edit_window_size * 2 + 1, attn_size])
    z = tf.nn.embedding_lookup(pos_embedding, i)

    v = get_variable_unsafe('v_a', [attn_size])
    s = f + y + z

    return tf.reduce_sum(v * tf.tanh(s), [2])


def global_attention(state, prev_weights, hidden_states, encoder, encoder_input_length, pos=None, scope=None,
                     **kwargs):
    with tf.variable_scope(scope or 'attention'):
        if encoder.edit_window_size is not None and encoder.edit_window_size >= 0:
            compute_energy_ = compute_energy_edits
        elif encoder.attention_filters > 0:
            compute_energy_ = compute_energy_with_filter
        else:
            compute_energy_ = compute_energy

        e = compute_energy_(
            hidden_states, state, prev_weights=prev_weights, attention_filters=encoder.attention_filters,
            attention_filter_length=encoder.attention_filter_length, attn_size=encoder.attn_size, pos=pos,
            edit_window_size=encoder.edit_window_size
        )
        e -= tf.reduce_max(e, axis=1, keep_dims=True)

        mask = tf.sequence_mask(tf.cast(encoder_input_length, tf.int32), tf.shape(hidden_states)[1],
                                dtype=tf.float32)
        exp = tf.exp(e) * mask
        weights = exp / tf.reduce_sum(exp, axis=-1, keep_dims=True)
        weighted_average = tf.reduce_sum(tf.expand_dims(weights, 2) * hidden_states, axis=1)

        return weighted_average, weights


def local_attention(state, prev_weights, hidden_states, encoder, encoder_input_length, pos=None, scope=None,
                    **kwargs):
    """
    Local attention of Luong et al. (http://arxiv.org/abs/1508.04025)
    """
    attn_length = tf.shape(hidden_states)[1]
    state_size = state.get_shape()[1].value

    with tf.variable_scope(scope or 'attention'):
        wp = get_variable_unsafe('Wp', [state_size, state_size])
        vp = get_variable_unsafe('vp', [state_size, 1])

        encoder_input_length = tf.cast(tf.expand_dims(encoder_input_length, axis=1), tf.float32)

        pos_ = pos
        if pos is None:
            pos = tf.nn.sigmoid(tf.matmul(tf.nn.tanh(tf.matmul(state, wp)), vp))
            pos = tf.floor(encoder_input_length * pos)

        pos = tf.reshape(pos, [-1, 1])

        batch_size = tf.shape(state)[0]

        idx = tf.tile(tf.cast(tf.range(attn_length), dtype=tf.float32), tf.stack([batch_size]))
        idx = tf.reshape(idx, [-1, attn_length])

        low = pos - encoder.attention_window_size
        high = pos + encoder.attention_window_size

        mlow = tf.to_float(idx < low)
        mhigh = tf.to_float(idx > high)
        # m = mlow + mhigh + tf.to_float(idx >= encoder_input_length)    # degrades performance

        # a = tf.to_float(idx >= encoder_input_length)
        # m = tf.Print(m, [idx[5]], message='idx', summarize=100)
        # m = tf.Print(m, [encoder_input_length[5]], message='length', summarize=100)
        # m = tf.Print(m, [a[5]], message='a', summarize=100)
        # m = tf.Print(m, [tf.reduce_sum(hidden_states[5], axis=1)], message='states', summarize=10000)

        m = mlow + mhigh
        mask = tf.to_float(tf.equal(m, 0.0))
        # mask_bis = tf.to_float(tf.equal(mlow + mhigh, 0.0))

        if encoder.edit_window_size is not None and encoder.edit_window_size >= 0:
            compute_energy_ = compute_energy_edits
        elif encoder.attention_filters > 0:
            compute_energy_ = compute_energy_with_filter
        else:
            compute_energy_ = compute_energy

        e = compute_energy_(
            hidden_states, state, prev_weights=prev_weights, attention_filters=encoder.attention_filters,
            attention_filter_length=encoder.attention_filter_length, attn_size=encoder.attn_size, pos=pos_,
            edit_window_size=encoder.edit_window_size
        )

        # import ipdb; ipdb.set_trace()
        # e = tf.Print(e, [e[5]], message='e', summarize=100)
        weights = softmax(e, mask=mask)
        # weights_bis = softmax(e, mask=mask_bis)
        # weights = tf.Print(weights, [mask[5]], message='mask', summarize=100)
        # weights = tf.Print(weights, [mask[5]], message='mask_bis', summarize=100)
        # mask = tf.to_float(tf.not_equal(tf.reduce_sum(hidden_states, axis=2), 0.0))[5]
        # weights = tf.Print(weights, [mask], message='true_mask', summarize=100)
        # weights = tf.Print(weights, [weights[5] * mask], message='weights', summarize=100)
        # weights = tf.Print(weights, [weights_bis[5] * mask], message='weights_bis', summarize=100)

        # import ipdb; ipdb.set_trace()

        sigma = encoder.attention_window_size / 2
        numerator = -tf.pow((idx - pos), tf.convert_to_tensor(2, dtype=tf.float32))
        # div = tf.truediv(numerator, 2 * sigma ** 2)
        div = tf.truediv(numerator, sigma ** 2)

        weights *= tf.exp(div)  # result of the truncated normal distribution
        # normalize to keep a probability distribution
        # weights /= (tf.reduce_sum(weights, axis=1, keep_dims=True) + 10e-12)

        weighted_average = tf.reduce_sum(tf.expand_dims(weights, axis=2) * hidden_states, axis=1)

        return weighted_average, weights


def attention(state, prev_weights, hidden_states, encoder, **kwargs):
    """
    Proxy for `local_attention` and `global_attention`
    """
    if encoder.attention_window_size > 0:
        attention_ = local_attention
    else:
        attention_ = global_attention

    return attention_(state, prev_weights, hidden_states, encoder, **kwargs)


def multi_attention(state, prev_weights, hidden_states, encoders, encoder_input_length, **kwargs):
    """
    Same as `attention` except that prev_weights, hidden_states and encoders
    are lists whose length is the number of encoders.
    """
    hidden_states = tf.unstack(hidden_states)

    attns, weights = list(zip(*[
        attention(state, weights, hidden, encoder, encoder_input_length=input_length,
                  scope='attention_{}'.format(encoder.name), **kwargs)
        for weights, hidden, encoder, input_length in zip(prev_weights, hidden_states, encoders, encoder_input_length)
    ]))

    return tf.concat(attns, 1), list(weights)


def get_embedding_function(decoder, encoders):
    if decoder.op_embedding_size is not None and decoder.op_embedding_size > 0:
        """
        OPS (<S>, </S>, <INS>, etc.)
        <INS>_w1
        ...
        <INS>_wn
        <SUB>_w1
        ...
        <SUB>_wn

        """
        op_count = len(utils._START_VOCAB)
        word_count = (decoder.vocab_size - op_count) // 2

        with tf.device('/cpu:0'):
            word_embedding_shape = [op_count + word_count, decoder.embedding_size]
            if encoders[0].name == decoder.name:
                word_embedding_shape[0] = encoders[0].vocab_size

            word_embedding = get_variable_unsafe('embedding_{}'.format(decoder.name), shape=word_embedding_shape)

        op_embedding_shape = [op_count, decoder.op_embedding_size]
        op_embedding = get_variable_unsafe('op_embedding', shape=op_embedding_shape)

        def embed(input_):
            def word_index(index):
                return tf.cond(index < op_count + word_count, lambda: index, lambda: index - word_count)
            def op_index(index):
                return tf.case([
                    (index >= op_count, lambda: tf.constant(utils.INS_ID, dtype=tf.int64)),
                    (index >= op_count + word_count, lambda: tf.constant(utils.SUB_ID, dtype=tf.int64))],
                default=lambda: index)

            op_indices = tf.map_fn(op_index, input_)
            word_indices = tf.map_fn(word_index, input_)

            op_embed = tf.nn.embedding_lookup(op_embedding, op_indices)
            word_embed = tf.nn.embedding_lookup(word_embedding, word_indices)

            return tf.concat([op_embed, word_embed], axis=1)

        return embed
    else:
        if decoder.get('embedding') is not None:
            initializer = decoder.embedding
            embedding_shape = None
        else:
            initializer = None
            # embedding_shape = [decoder.oracle_vocab_size or decoder.vocab_size, decoder.embedding_size]
            embedding_shape = [decoder.vocab_size, decoder.embedding_size]

        with tf.device('/cpu:0'):
            embedding = get_variable_unsafe('embedding_{}'.format(decoder.name), shape=embedding_shape,
                                            initializer=initializer)
        def embed(input_):
            if embedding is not None:
                return tf.nn.embedding_lookup(embedding, input_)
            else:
                return input_

        return embed


def attention_decoder_old(decoder_inputs, initial_state, attention_states, encoders, decoder, encoder_input_length,
                      decoder_input_length=None, dropout=None, feed_previous=0.0, feed_argmax=True, use_edits=False,
                      **kwargs):
    """
    :param targets: tensor of shape (output_length, batch_size)
    :param initial_state: initial state of the decoder (usually the final state of the encoder),
      as a tensor of shape (batch_size, initial_state_size). This state is mapped to the
      correct state size for the decoder.
    :param attention_states: list of tensors of shape (batch_size, input_length, encoder_cell_size),
      usually the encoder outputs (one tensor for each encoder).
    :param encoders: configuration of the encoders
    :param decoder: configuration of the decoder
    :param decoder_input_length:
    :param dropout: scalar tensor or None, specifying the keep probability (1 - dropout)
    :param feed_previous: scalar tensor corresponding to the probability to use previous decoder output
      instead of the groundtruth as input for the decoder (1 when decoding, between 0 and 1 when training)
    :return:
      outputs of the decoder as a tensor of shape (batch_size, output_length, decoder_cell_size)
      attention weights as a tensor of shape (output_length, encoders, batch_size, input_length)
    """
    # TODO: dropout instead of keep probability
    assert decoder.cell_size % 2 == 0, 'cell size must be a multiple of 2'   # because of maxout

    embed = get_embedding_function(decoder, encoders)

    if decoder.use_lstm:
        cell = BasicLSTMCell(decoder.cell_size, state_is_tuple=False)
    else:
        cell = GRUCell(decoder.cell_size, initializer=orthogonal_initializer())

    if dropout is not None:
        cell = DropoutWrapper(cell, input_keep_prob=dropout)

    if decoder.layers > 1:
        cell = MultiRNNCell([cell] * decoder.layers, residual_connections=decoder.residual_connections)

    with tf.variable_scope('decoder_{}'.format(decoder.name)):
        attention_ = functools.partial(multi_attention, hidden_states=attention_states, encoders=encoders,
                                       encoder_input_length=encoder_input_length)
        input_shape = tf.shape(decoder_inputs)
        time_steps = input_shape[0]
        batch_size = input_shape[1]

        if decoder.oracle:
            output_size = len(utils._START_VOCAB)
        else:
            output_size = decoder.vocab_size

        state_size = cell.state_size

        if initial_state is not None:
            if dropout is not None:
                initial_state = tf.nn.dropout(initial_state, dropout)

            state = tf.nn.tanh(
                linear_unsafe(initial_state, state_size, True, scope='initial_state_projection')
            )
        else:
            # if not initial state, initialize with zeroes (this is the case for MIXER)
            state = tf.zeros([batch_size, state_size], dtype=tf.float32)

        sequence_length = decoder_input_length
        if sequence_length is not None:
            sequence_length = tf.to_int32(sequence_length)
            min_sequence_length = tf.reduce_min(sequence_length)
            max_sequence_length = tf.reduce_max(sequence_length)

        time = tf.constant(0, dtype=tf.int32, name='time')
        zero_output = tf.zeros(tf.stack([batch_size, cell.output_size]), tf.float32)

        proj_outputs = tf.TensorArray(dtype=tf.float32, size=time_steps, clear_after_read=False)
        decoder_outputs = tf.TensorArray(dtype=tf.float32, size=time_steps)

        inputs = tf.TensorArray(dtype=tf.int64, size=time_steps, clear_after_read=False).unstack(
                                tf.cast(decoder_inputs, tf.int64))
        samples = tf.TensorArray(dtype=tf.int64, size=time_steps, clear_after_read=False)
        states = tf.TensorArray(dtype=tf.float32, size=time_steps)

        attn_lengths = [tf.shape(states)[1] for states in attention_states]

        weights = tf.TensorArray(dtype=tf.float32, size=time_steps)
        initial_weights = [tf.zeros(tf.stack([batch_size, length])) for length in attn_lengths]

        output = tf.zeros(tf.stack([batch_size, cell.output_size]), dtype=tf.float32)

        initial_input = embed(inputs.read(0))   # first symbol is BOS

        edit_pos = tf.zeros([batch_size], tf.int32)

        def _time_step(time, input_, state, output, proj_outputs, decoder_outputs, samples, states, weights,
                       prev_weights, edit_pos):
            pos = None
            if use_edits:
                pos = tf.cast(edit_pos, tf.float32)

            context_vector, new_weights = attention_(state, prev_weights=prev_weights, pos=pos)

            weights = weights.write(time, new_weights)

            # FIXME use `output` or `state` here?
            output_ = linear_unsafe([state, input_, context_vector], decoder.cell_size, False, scope='maxout')
            output_ = tf.reduce_max(tf.reshape(output_, tf.stack([batch_size, decoder.cell_size // 2, 2])), axis=2)
            output_ = linear_unsafe(output_, decoder.embedding_size, False, scope='softmax0')
            decoder_outputs = decoder_outputs.write(time, output_)
            output_ = linear_unsafe(output_, output_size, True, scope='softmax1')
            proj_outputs = proj_outputs.write(time, output_)

            argmax = lambda: tf.argmax(output_, 1)
            softmax = lambda: tf.squeeze(tf.multinomial(tf.log(tf.nn.softmax(output_)), num_samples=1),
                                         axis=1)
            target = lambda: inputs.read(time + 1)

            sample = tf.case([
                (tf.logical_and(time < time_steps - 1, tf.random_uniform([]) >= feed_previous), target),
                (tf.logical_not(feed_argmax), softmax)],
                default=argmax)   # default case is useful for beam-search

            sample.set_shape([None])
            sample = tf.stop_gradient(sample)   # apply oracle here

            is_keep = tf.equal(sample, utils.KEEP_ID)
            is_sub = tf.equal(sample, utils.SUB_ID)
            is_del = tf.equal(sample, utils.DEL_ID)

            i = tf.logical_or(is_keep, is_sub)
            i = tf.logical_or(i, is_del)
            i = tf.to_int32(i)
            edit_pos += i

            # sample = tf.Print(sample, [time, time_steps])
            samples = samples.write(time, sample)

            input_ = embed(sample)

            x = tf.concat([input_, context_vector], 1)
            call_cell = lambda: unsafe_decorator(cell)(x, state)

            if sequence_length is not None:
                new_output, new_state = rnn._rnn_step(
                    time=time,
                    sequence_length=sequence_length,
                    min_sequence_length=min_sequence_length,
                    max_sequence_length=max_sequence_length,
                    zero_output=zero_output,
                    state=state,
                    call_cell=call_cell,
                    state_size=state_size,
                    skip_conditionals=True)
            else:
                new_output, new_state = call_cell()

            states = states.write(time, new_state)

            return (time + 1, input_, new_state, new_output, proj_outputs, decoder_outputs, samples, states, weights,
                    new_weights, edit_pos)

        (_, _, new_state, new_output, proj_outputs, decoder_outputs, samples, states, weights, _, new_edit_pos
         ) = tf.while_loop(
            cond=lambda time, *_: time < time_steps,
            body=_time_step,
            loop_vars=(time, initial_input, state, output, proj_outputs, decoder_outputs, samples, weights, states,
                       initial_weights, edit_pos),
            parallel_iterations=decoder.parallel_iterations,
            swap_memory=decoder.swap_memory)

        proj_outputs = proj_outputs.stack()
        decoder_outputs = decoder_outputs.stack()
        samples = samples.stack()
        weights = weights.stack()  # batch_size, encoders, output time, input time
        states = states.stack()

        # weights = tf.Print(weights, [weights[:,0]], summarize=20)
        # tf.control_dependencies()

        beam_tensors = namedtuple('beam_tensors', 'state new_state output new_output edit_pos new_edit_pos')
        return (proj_outputs, weights, decoder_outputs, beam_tensors(state, new_state, output, new_output,
                                                                     edit_pos, new_edit_pos),
                samples, states)


def attention_decoder(decoder_inputs, initial_state, attention_states, encoders, decoder, encoder_input_length,
                      decoder_input_length=None, feed_previous=0.0, dropout=None, use_edits=False, **kwargs):
    assert decoder.cell_size % 2 == 0, 'cell size must be a multiple of 2'   # because of maxout

    embed = get_embedding_function(decoder, encoders)

    if decoder.use_lstm:
        cell = BasicLSTMCell(decoder.cell_size, state_is_tuple=False)
    else:
        cell = GRUCell(decoder.cell_size, initializer=orthogonal_initializer())

    if dropout is not None:
        cell = DropoutWrapper(cell, input_keep_prob=dropout)

    if decoder.layers > 1:
        cell = MultiRNNCell([cell] * decoder.layers, residual_connections=decoder.residual_connections)

    with tf.variable_scope('decoder_{}'.format(decoder.name)):
        attention_ = functools.partial(multi_attention, hidden_states=attention_states, encoders=encoders,
                                       encoder_input_length=encoder_input_length, prev_weights=[None])
        input_shape = tf.shape(decoder_inputs)
        time_steps = input_shape[0]
        batch_size = input_shape[1]

        if decoder.oracle:
            output_size = len(utils._START_VOCAB)
        else:
            output_size = decoder.vocab_size

        state_size = cell.state_size

        state = tf.nn.tanh(linear_unsafe(initial_state, state_size, True, scope='initial_state_projection'))

        sequence_length = tf.to_int32(decoder_input_length)
        min_sequence_length = tf.reduce_min(sequence_length)
        max_sequence_length = tf.reduce_max(sequence_length)

        time = tf.constant(0, dtype=tf.int32, name='time')
        zero_output = tf.zeros(tf.stack([batch_size, cell.output_size]), tf.float32)

        outputs = tf.TensorArray(dtype=tf.float32, size=time_steps, clear_after_read=False)

        inputs = tf.TensorArray(dtype=tf.int64, size=time_steps, clear_after_read=False).unstack(
                                tf.cast(decoder_inputs, tf.int64))

        initial_input = embed(inputs.read(0))   # first symbol is BOS

        edit_pos = tf.zeros([batch_size], tf.int32)

        def get_output(input_, state, pos=None):
            context_vector, _ = attention_(state, pos=pos)
            output_ = linear_unsafe([state, input_, context_vector], decoder.cell_size, False, scope='maxout')
            output_ = tf.reduce_max(tf.reshape(output_, tf.stack([batch_size, decoder.cell_size // 2, 2])), axis=2)
            output_ = linear_unsafe(output_, decoder.embedding_size, False, scope='softmax0')
            proj_output = linear_unsafe(output_, output_size, True, scope='softmax1')
            return context_vector, proj_output

        pos = None
        if use_edits:
            pos = tf.cast(edit_pos, tf.float32)

        context_vector, output = get_output(initial_input, state, pos=pos)
        outputs = outputs.write(time, output)
        time += 1

        def _time_step(time, state, output, outputs, context_vector, edit_pos):
            argmax = lambda: tf.argmax(output, axis=1)
            target = lambda: inputs.read(time)
            sample = tf.cond(tf.random_uniform([]) >= feed_previous, target, argmax)
            sample.set_shape([None])
            sample = tf.stop_gradient(sample)
            input_ = embed(sample)

            is_keep = tf.equal(sample, utils.KEEP_ID)
            is_sub = tf.equal(sample, utils.SUB_ID)
            is_del = tf.equal(sample, utils.DEL_ID)

            i = tf.logical_or(is_keep, is_sub)
            i = tf.logical_or(i, is_del)
            i = tf.to_int32(i)
            edit_pos += i

            x = tf.concat([input_, context_vector], 1)
            call_cell = lambda: unsafe_decorator(cell)(x, state)
            _, state = rnn._rnn_step(
                time=time,
                sequence_length=sequence_length,
                min_sequence_length=min_sequence_length,
                max_sequence_length=max_sequence_length,
                zero_output=zero_output,
                state=state,
                call_cell=call_cell,
                state_size=state_size,
                skip_conditionals=True)

            pos = None
            if use_edits:
                pos = tf.cast(edit_pos, tf.float32)

            context_vector, output = get_output(input_, state, pos=pos)
            outputs = outputs.write(time, output)
            return time + 1, state, output, outputs, context_vector, edit_pos

        new_time, new_state, new_output, outputs, new_context_vector, new_edit_pos = tf.while_loop(
            cond=lambda time, *_: time < time_steps,
            body=_time_step,
            loop_vars=(time, state, output, outputs, context_vector, edit_pos),
            parallel_iterations=decoder.parallel_iterations,
            swap_memory=decoder.swap_memory)

        outputs = outputs.stack()
        return outputs


def beam_attention_decoder(initial_state, attention_states, encoders, decoder, encoder_input_length,
                           use_edits=False, **kwargs):
    assert decoder.cell_size % 2 == 0, 'cell size must be a multiple of 2'   # because of maxout

    embed = get_embedding_function(decoder, encoders)
    if decoder.use_lstm:
        cell = BasicLSTMCell(decoder.cell_size, state_is_tuple=False)
    else:
        cell = GRUCell(decoder.cell_size, initializer=orthogonal_initializer())
    if decoder.layers > 1:
        cell = MultiRNNCell([cell] * decoder.layers, residual_connections=decoder.residual_connections)

    with tf.variable_scope('decoder_{}'.format(decoder.name)):
        attention_ = functools.partial(multi_attention, hidden_states=attention_states, encoders=encoders,
                                       encoder_input_length=encoder_input_length, prev_weights=[None])
        if decoder.oracle:
            output_size = len(utils._START_VOCAB)
        else:
            output_size = decoder.vocab_size

        state_size = cell.state_size
        initial_state = tf.nn.tanh(linear_unsafe(initial_state, state_size, True, scope='initial_state_projection'))

        def get_output(input_, state, pos=None):
            context_vector, _ = attention_(state, pos=pos)
            output_ = linear_unsafe([state, input_, context_vector], decoder.cell_size, False, scope='maxout')
            output_ = tf.reduce_max(tf.reshape(output_, tf.stack([batch_size, decoder.cell_size // 2, 2])), axis=2)
            output_ = linear_unsafe(output_, decoder.embedding_size, False, scope='softmax0')
            proj_output = linear_unsafe(output_, output_size, True, scope='softmax1')
            return context_vector, proj_output

        batch_size = tf.shape(initial_state)[0]

        initial_edit_pos = tf.zeros([batch_size, 1], tf.float32)
        initial_input = embed(tf.ones(shape=[batch_size], dtype=tf.int32) * utils.BOS_ID)

        initial_context_vector, initial_output = get_output(initial_input, initial_state, pos=initial_edit_pos)

        state_size = initial_state.get_shape()[1]
        context_size = initial_context_vector.get_shape()[1]

        data = tf.placeholder(tf.float32, shape=[None, state_size + context_size + 1])

        state, context_vector, edit_pos = tf.split(data, [state_size.value, context_size.value, 1], axis=1)
        initial_data = tf.concat([initial_state, initial_context_vector, initial_edit_pos], axis=1)
        decoder_input = tf.placeholder(tf.int64, shape=[None])

        if use_edits:
            is_keep = tf.equal(decoder_input, utils.KEEP_ID)
            is_sub = tf.equal(decoder_input, utils.SUB_ID)
            is_del = tf.equal(decoder_input, utils.DEL_ID)

            i = tf.logical_or(is_keep, is_sub)
            i = tf.logical_or(i, is_del)
            i = tf.to_float(i)
            new_edit_pos = edit_pos + tf.expand_dims(i, axis=1)
        else:
            new_edit_pos = edit_pos

        embedded_input = embed(decoder_input)
        x = tf.concat([embedded_input, context_vector], axis=1)
        call_cell = lambda: unsafe_decorator(cell)(x, state)
        _, new_state = call_cell()
        new_context_vector, new_output = get_output(embedded_input, new_state, pos=new_edit_pos)
        new_data = tf.concat([new_state, new_context_vector, new_edit_pos], axis=1)

        beam_tensors = utils.AttrDict(
            initial_output=tf.nn.softmax(initial_output),
            output=tf.nn.softmax(new_output),
            decoder_input=decoder_input,
            initial_data=initial_data,
            data=data,
            new_data=new_data,
        )

        return beam_tensors


def sequence_loss(logits, targets, weights, average_across_timesteps=False, average_across_batch=True,
                  reward=None):
    time_steps = tf.shape(targets)[0]
    batch_size = tf.shape(targets)[1]

    logits_ = tf.reshape(logits, tf.stack([time_steps * batch_size, logits.get_shape()[2].value]))
    targets_ = tf.reshape(targets, tf.stack([time_steps * batch_size]))

    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_, labels=targets_)
    crossent = tf.reshape(crossent, tf.stack([time_steps, batch_size]))

    if reward is not None:
        crossent *= tf.stop_gradient(reward)

    log_perp = tf.reduce_sum(crossent * weights, 0)

    if average_across_timesteps:
        total_size = tf.reduce_sum(weights, 0)
        total_size += 1e-12  # just to avoid division by 0 for all-0 weights
        log_perp /= total_size

    cost = tf.reduce_sum(log_perp)

    if average_across_batch:
        batch_size = tf.shape(targets)[1]
        return cost / tf.cast(batch_size, tf.float32)
    else:
        return cost


def baseline_loss(reward, weights, average_across_timesteps=False,
                  average_across_batch=True):
    """
    :param reward: tensor of shape (time_steps, batch_size)
    :param weights: tensor of shape (time_steps, batch_size)
    """
    batch_size = tf.shape(reward)[1]

    cost = reward ** 2
    cost = tf.reduce_sum(cost * weights, axis=0)

    if average_across_timesteps:
        total_size = tf.reduce_sum(weights, 0)
        total_size += 1e-12  # just to avoid division by 0 for all-0 weights
        cost /= total_size

    cost = tf.reduce_sum(cost)

    if average_across_batch:
        cost /= tf.cast(batch_size, tf.float32)

    return cost


def reinforce_baseline(decoder_states, reward):
    """
    Center the reward by computing a baseline reward over decoder states.

    :param decoder_states: internal states of the decoder, tensor of shape (time_steps, batch_size, state_size)
    :param reward: reward for each time step, tensor of shape (time_steps, batch_size)
    :return: reward - computed baseline, tensor of shape (time_steps, batch_size)
    """
    time_steps = tf.shape(decoder_states)[0]
    batch_size = tf.shape(decoder_states)[1]
    state_size = decoder_states.get_shape()[2]

    states = tf.reshape(decoder_states, shape=tf.stack([time_steps * batch_size, state_size]))

    baseline = fully_connected(tf.stop_gradient(states), num_outputs=1, activation_fn=None,
                               scope='reward_baseline',
                               weights_initializer=tf.constant_initializer(0.0),
                               biases_initializer=tf.constant_initializer(0.01))

    baseline = tf.reshape(baseline, shape=tf.stack([time_steps, batch_size]))

    return reward - baseline


def softmax(logits, dim=-1, temperature=1.0, mask=None):
    """
    Softmax with a `temperature` parameter:
        - 1.0: same as tf.nn.softmax
        - < 0: closer to argmax
        - > 1: closer to uniform distribution
    """
    # TODO: corresponding loss
    e = tf.exp(logits / temperature)
    if mask is not None:
        e *= mask

    return e / (tf.reduce_sum(e, axis=dim, keep_dims=True) + 10e-12)  # FIXME


def get_weights(sequence, eos_id, time_major=False, include_first_eos=True):
    axis = 1 - time_major

    weights = (1.0 - tf.minimum(
        tf.cumsum(tf.cast(tf.equal(sequence, eos_id), tf.float32), axis=axis), 1.0))

    if include_first_eos:
        weights = weights[:-1,:] if time_major else weights[:,:-1]
        shape = [tf.shape(weights)[0], tf.shape(weights)[1]]
        shape[axis] = 1
        weights = tf.concat([tf.ones(tf.stack(shape)), weights], axis)

    return tf.stop_gradient(weights)
