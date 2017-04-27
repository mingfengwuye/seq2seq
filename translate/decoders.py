import tensorflow as tf
import functools
from tensorflow.contrib.rnn import BasicLSTMCell, DropoutWrapper, LayerNormBasicLSTMCell
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn, MultiRNNCell, LSTMStateTuple, GRUCell
from translate import utils


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


get_variable_unsafe = unsafe_decorator(tf.get_variable)


def multi_encoder(encoder_inputs, encoders, encoder_input_length, other_inputs=None, dropout=None, **kwargs):
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
        embedding_shape = [encoder.vocab_size, encoder.embedding_size]
        with tf.device('/cpu:0'):
            embedding = get_variable_unsafe('embedding_{}'.format(encoder.name), shape=embedding_shape)
        embedding_variables.append(embedding)

    for i, encoder in enumerate(encoders):
        with tf.variable_scope('encoder_{}'.format(encoder.name)):
            encoder_inputs_ = encoder_inputs[i]
            encoder_input_length_ = encoder_input_length[i]

            def get_cell():
                if encoder.use_lstm:
                    keep_prob = dropout if dropout else 1.0
                    cell = LayerNormBasicLSTMCell(encoder.cell_size, dropout_keep_prob=keep_prob,
                                                  layer_norm=encoder.layer_norm)
                else:
                    cell = GRUCell(encoder.cell_size)

                    if dropout is not None:
                        cell = DropoutWrapper(cell, input_keep_prob=dropout)

                return cell

            embedding = embedding_variables[i]

            if embedding is not None:
                batch_size = tf.shape(encoder_inputs_)[0]
                time_steps = tf.shape(encoder_inputs_)[1]

                flat_inputs = tf.reshape(encoder_inputs_, [tf.multiply(batch_size, time_steps)])
                flat_inputs = tf.nn.embedding_lookup(embedding, flat_inputs)
                encoder_inputs_ = tf.reshape(flat_inputs,
                                             tf.stack([batch_size, time_steps, flat_inputs.get_shape()[1].value]))

            if other_inputs is not None:
                encoder_inputs_ = tf.concat([encoder_inputs_, other_inputs], axis=2)

            # Contrary to Theano's RNN implementation, states after the sequence length are zero
            # (while Theano repeats last state)
            parameters = dict(
                inputs=encoder_inputs_, sequence_length=encoder_input_length_,
                dtype=tf.float32, parallel_iterations=encoder.parallel_iterations,
            )

            state_size = get_cell().state_size
            if isinstance(state_size, LSTMStateTuple):
                state_size = state_size.c + state_size.h

            def get_initial_state(name='initial_state'):
                initial_state = tf.get_variable(name, initializer=tf.zeros(state_size))
                initial_state = tf.tile(tf.expand_dims(initial_state, axis=0), [batch_size, 1])
                if isinstance(get_cell().state_size, LSTMStateTuple):
                    return LSTMStateTuple(*tf.split(initial_state, 2, axis=1))
                else:
                    return initial_state

            if encoder.bidir:
                encoder_outputs_, _, _ = stack_bidirectional_dynamic_rnn(
                    cells_fw=[get_cell() for _ in range(encoder.layers)],
                    cells_bw=[get_cell() for _ in range(encoder.layers)],
                    initial_states_fw=[get_initial_state('initial_state_fw')] * encoder.layers,
                    initial_states_bw=[get_initial_state('initial_state_bw')] * encoder.layers,
                    **parameters
                )
                encoder_state_ = encoder_outputs_[:, 0, encoder.cell_size:]  # first backward output
            else:
                if encoder.layers > 1:
                    cell = MultiRNNCell([get_cell() for _ in range(encoder.layers)])
                else:
                    cell = get_cell()

                encoder_outputs_, _ = tf.nn.dynamic_rnn(cell=cell, initial_state=get_initial_state(), **parameters)
                encoder_state_ = encoder_outputs_[:, -1, :]  # FIXME

            encoder_outputs.append(encoder_outputs_)
            encoder_states.append(encoder_state_)

    encoder_state = tf.concat(encoder_states, 1)
    return encoder_outputs, encoder_state


def compute_energy(hidden, state, attn_size, **kwargs):
    input_size = hidden.get_shape()[2].value

    y = tf.layers.dense(state, attn_size, use_bias=True, name='W_a')
    y = tf.expand_dims(y, axis=1)

    k = tf.get_variable('U_a', [input_size, attn_size])
    f = tf.einsum('ijk,kl->ijl', hidden, k)

    v = tf.get_variable('v_a', [attn_size])
    s = f + y

    return tf.reduce_sum(v * tf.tanh(s), [2])


def global_attention(state, hidden_states, encoder, encoder_input_length, pos=None, scope=None,
                     context=None, **kwargs):
    with tf.variable_scope(scope or 'attention'):
        if context is not None and encoder.use_context:
            state = tf.concat([state, context], axis=1)

        e = compute_energy(hidden_states, state, attn_size=encoder.attn_size, pos=pos)
        e -= tf.reduce_max(e, axis=1, keep_dims=True)

        mask = tf.sequence_mask(tf.cast(encoder_input_length, tf.int32), tf.shape(hidden_states)[1],
                                dtype=tf.float32)
        exp = tf.exp(e) * mask
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
    """
    Local attention of Luong et al. (http://arxiv.org/abs/1508.04025)
    """
    batch_size = tf.shape(state)[0]
    attn_length = tf.shape(hidden_states)[1]

    if context is not None and encoder.use_context:
        state = tf.concat([state, context], axis=1)

    state_size = state.get_shape()[1].value

    with tf.variable_scope(scope or 'attention'):
        encoder_input_length = tf.to_float(tf.expand_dims(encoder_input_length, axis=1))

        pos_ = pos
        if pos is None:
            wp = tf.get_variable('Wp', [state_size, state_size])
            vp = tf.get_variable('vp', [state_size, 1])

            pos = tf.nn.sigmoid(tf.matmul(tf.nn.tanh(tf.matmul(state, wp)), vp))
            pos = tf.floor(encoder_input_length * pos)

        pos = tf.reshape(pos, [-1, 1])
        # pos = tf.minimum(pos, encoder_input_length - 1)

        if encoder.attention_window_size == 0:
            weights = tf.to_float(tf.one_hot(tf.cast(tf.squeeze(pos, axis=1), tf.int32), depth=attn_length))
        else:
            idx = tf.tile(tf.to_float(tf.range(attn_length)), tf.stack([batch_size]))
            idx = tf.reshape(idx, [-1, attn_length])

            low = pos - encoder.attention_window_size
            high = pos + encoder.attention_window_size

            mlow = tf.to_float(idx < low)
            mhigh = tf.to_float(idx > high)
            m = mlow + mhigh
            m += tf.to_float(idx >= encoder_input_length)  # seems to degrades performance

            mask = tf.to_float(tf.equal(m, 0.0))

            e = compute_energy(hidden_states, state, attn_size=encoder.attn_size, pos=pos_)

            weights = softmax(e, mask=mask)

            sigma = encoder.attention_window_size / 2
            numerator = -tf.pow((idx - pos), tf.convert_to_tensor(2, dtype=tf.float32))
            div = tf.truediv(numerator, 2 * sigma ** 2)
            # div = tf.truediv(numerator, sigma ** 2)
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
                    **kwargs):
    attns = []
    weights = []

    context_vector = None
    for i, (hidden, encoder, input_length) in enumerate(zip(hidden_states, encoders, encoder_input_length)):
        pos_ = pos[i] if pos is not None else None
        context_vector, weights_ = attention(state=state, hidden_states=hidden, encoder=encoder,
                                             encoder_input_length=input_length, pos=pos_, context=context_vector,
                                             **kwargs)
        attns.append(context_vector)
        weights.append(weights_)

    if aggregation_method == 'sum':
        context_vector = tf.reduce_sum(tf.stack(attns, axis=2), axis=2)
    else:
        context_vector = tf.concat(attns, axis=1)

    return context_vector, weights


def get_embedding_function(decoder):
    embedding_shape = [decoder.vocab_size, decoder.embedding_size]
    with tf.device('/cpu:0'):
        embedding = get_variable_unsafe('embedding_{}'.format(decoder.name), shape=embedding_shape)

    def embed(input_):
        return tf.nn.embedding_lookup(embedding, input_)
    return embed


def attention_decoder(decoder_inputs, initial_state, attention_states, encoders, decoder, encoder_input_length,
                      dropout=None, feed_previous=0.0, align_encoder_id=0, **kwargs):
    """
    :param targets: tensor of shape (output_length, batch_size)
    :param initial_state: initial state of the decoder (usually the final state of the encoder),
      as a tensor of shape (batch_size, initial_state_size). This state is mapped to the
      correct state size for the decoder.
    :param attention_states: list of tensors of shape (batch_size, input_length, encoder_cell_size),
      usually the encoder outputs (one tensor for each encoder).
    :param encoders: configuration of the encoders
    :param decoder: configuration of the decoder
    :param dropout: scalar tensor or None, specifying the keep probability (1 - dropout)
    :param feed_previous: scalar tensor corresponding to the probability to use previous decoder output
      instead of the groundtruth as input for the decoder (1 when decoding, between 0 and 1 when training)
    :return:
      outputs of the decoder as a tensor of shape (batch_size, output_length, decoder_cell_size)
      attention weights as a tensor of shape (output_length, encoders, batch_size, input_length)
    """
    # TODO: dropout instead of keep probability
    assert decoder.cell_size % 2 == 0, 'cell size must be a multiple of 2'   # because of maxout

    embed = get_embedding_function(decoder)

    def get_cell():
        if decoder.use_lstm:
            keep_prob = dropout if dropout else 1.0
            cell = LayerNormBasicLSTMCell(decoder.cell_size, dropout_keep_prob=keep_prob,
                                          layer_norm=decoder.layer_norm)
        else:
            cell = GRUCell(decoder.cell_size)

            if dropout is not None:
                cell = DropoutWrapper(cell, input_keep_prob=dropout)

        return cell

    if decoder.layers > 1:
        cell = MultiRNNCell([get_cell() for _ in range(decoder.layers)])
    else:
        cell = get_cell()

    with tf.variable_scope('decoder_{}'.format(decoder.name)):
        attention_ = functools.partial(multi_attention, hidden_states=attention_states, encoders=encoders,
                                       encoder_input_length=encoder_input_length,
                                       aggregation_method=decoder.aggregation_method)
        input_shape = tf.shape(decoder_inputs)
        batch_size = input_shape[0]
        time_steps = input_shape[1]

        output_size = decoder.vocab_size

        state_size = get_cell().state_size
        if isinstance(state_size, LSTMStateTuple):
            state_size = state_size.c + state_size.h

        if dropout is not None:
            initial_state = tf.nn.dropout(initial_state, dropout)

        state = tf.layers.dense(initial_state, state_size, use_bias=True, name='initial_state_projection',
                                activation=tf.nn.tanh)

        edit_pos = tf.zeros([batch_size], tf.float32)

        # used by beam-search decoder (by dict feeding)
        data = tf.concat([state, tf.expand_dims(edit_pos, axis=1)], axis=1)
        state, edit_pos = tf.split(data, [state_size, 1], axis=1)
        edit_pos = tf.squeeze(edit_pos, axis=1)

        time = tf.constant(0, dtype=tf.int32, name='time')
        proj_outputs = tf.TensorArray(dtype=tf.float32, size=time_steps, clear_after_read=False)
        decoder_outputs = tf.TensorArray(dtype=tf.float32, size=time_steps)

        inputs = tf.TensorArray(dtype=tf.int64, size=time_steps, clear_after_read=False).unstack(
                                tf.cast(tf.transpose(decoder_inputs, perm=(1, 0)), tf.int64))
        states = tf.TensorArray(dtype=tf.float32, size=time_steps)
        weights = tf.TensorArray(dtype=tf.float32, size=time_steps)
        attns = tf.TensorArray(dtype=tf.float32, size=time_steps)

        initial_input = embed(inputs.read(0))   # first symbol is BOS

        def _time_step(time, input_, state, proj_outputs, decoder_outputs, states, weights, edit_pos, attns):
            pos = [edit_pos if encoder.align_edits else None for encoder in encoders]

            context_vector, new_weights = attention_(state, pos=pos)
            attns = attns.write(time, context_vector)

            weights = weights.write(time, new_weights[align_encoder_id])

            # FIXME use `output` or `state` here?
            x = tf.concat([state, input_, context_vector], axis=1)
            output_ = tf.layers.dense(x, decoder.cell_size, use_bias=False, name='maxout')
            output_ = tf.reduce_max(tf.reshape(output_, tf.stack([batch_size, decoder.cell_size // 2, 2])), axis=2)
            output_ = tf.layers.dense(output_, decoder.embedding_size, use_bias=False, name='softmax0')
            decoder_outputs = decoder_outputs.write(time, output_)
            output_ = tf.layers.dense(output_, output_size, use_bias=True, name='softmax1')

            proj_outputs = proj_outputs.write(time, output_)

            argmax = lambda: tf.argmax(output_, 1)
            target = lambda: inputs.read(time + 1)

            sample = tf.cond(tf.logical_and(time < time_steps - 1, tf.random_uniform([]) >= feed_previous),
                             target, argmax)
            sample.set_shape([None])
            sample = tf.stop_gradient(sample)

            if any(encoder.align_edits for encoder in encoders):
                is_keep = tf.equal(sample, utils.KEEP_ID)
                is_sub = tf.equal(sample, utils.SUB_ID)
                is_del = tf.equal(sample, utils.DEL_ID)

                i = tf.logical_or(is_keep, is_sub)
                i = tf.logical_or(i, is_del)
                i = tf.to_float(i)
                edit_pos += i

            input_ = embed(sample)

            x = tf.concat([input_, context_vector], 1)

            if isinstance(cell.state_size, LSTMStateTuple):
                state = LSTMStateTuple(*tf.split(state, 2, axis=1))

            _, new_state = cell(x, state)

            if isinstance(new_state, LSTMStateTuple):
                new_state = tf.concat([new_state.c, new_state.h], axis=1)

            states = states.write(time, new_state)

            return time + 1, input_, new_state, proj_outputs, decoder_outputs, states, weights, edit_pos, attns

        _, _, new_state, proj_outputs, decoder_outputs, states, weights, new_edit_pos, attns = tf.while_loop(
            cond=lambda time, *_: time < time_steps,
            body=_time_step,
            loop_vars=(time, initial_input, state, proj_outputs, decoder_outputs, weights, states, edit_pos, attns),
            parallel_iterations=decoder.parallel_iterations,
            swap_memory=decoder.swap_memory)

        proj_outputs = proj_outputs.stack()
        decoder_outputs = decoder_outputs.stack()
        weights = weights.stack()  # batch_size, encoders, output time, input time
        states = states.stack()
        attns = attns.stack()

        new_data = tf.concat([new_state, tf.expand_dims(new_edit_pos, axis=1)], axis=1)

        beam_tensors = utils.AttrDict(data=data, new_data=new_data)

        proj_outputs = tf.transpose(proj_outputs, perm=(1, 0, 2))
        weights = tf.transpose(weights, perm=(1, 0, 2))
        decoder_outputs = tf.transpose(decoder_outputs, perm=(1, 0, 2))
        states = tf.transpose(states, perm=(1, 0, 2))
        attns = tf.transpose(attns, perm=(1, 0, 2))

        return proj_outputs, weights, decoder_outputs, states, attns, beam_tensors


def sequence_loss(logits, targets, weights, average_across_timesteps=False, average_across_batch=True):
    batch_size = tf.shape(targets)[0]
    time_steps = tf.shape(targets)[1]

    logits_ = tf.reshape(logits, tf.stack([time_steps * batch_size, logits.get_shape()[2].value]))
    targets_ = tf.reshape(targets, tf.stack([time_steps * batch_size]))

    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_, labels=targets_)
    crossent = tf.reshape(crossent, tf.stack([batch_size, time_steps]))

    log_perp = tf.reduce_sum(crossent * weights, axis=1)

    if average_across_timesteps:
        total_size = tf.reduce_sum(weights, 0)
        total_size += 1e-12  # just to avoid division by 0 for all-0 weights
        log_perp /= total_size

    cost = tf.reduce_sum(log_perp)

    if average_across_batch:
        return cost / tf.cast(batch_size, tf.float32)
    else:
        return cost


def softmax(logits, dim=-1, mask=None):
    e = tf.exp(logits)
    if mask is not None:
        e *= mask

    return e / (tf.reduce_sum(e, axis=dim, keep_dims=True) + 10e-12)


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
