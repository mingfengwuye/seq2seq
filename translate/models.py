import tensorflow as tf
import functools
from tensorflow.contrib.rnn import BasicLSTMCell, DropoutWrapper, LayerNormBasicLSTMCell, RNNCell
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn, MultiRNNCell, LSTMStateTuple, GRUCell
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


def multi_encoder(encoder_inputs, encoders, encoder_input_length, other_inputs=None, dropout=None,
                  **kwargs):
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
    encoder_states = []
    encoder_outputs = []

    # create embeddings in the global scope (allows sharing between encoder and decoder)
    embedding_variables = []
    for encoder in encoders:
        # inputs are token ids, which need to be mapped to vectors (embeddings)
        embedding_shape = [encoder.vocab_size, encoder.embedding_size]
        with tf.device('/cpu:0'):
            embedding = get_variable('embedding_{}'.format(encoder.name), shape=embedding_shape)
        embedding_variables.append(embedding)

    for i, encoder in enumerate(encoders):
        with tf.variable_scope('encoder_{}'.format(encoder.name)):
            encoder_inputs_ = encoder_inputs[i]
            encoder_input_length_ = encoder_input_length[i]

            def get_cell(reuse=False):
                if encoder.use_lstm:
                    keep_prob = dropout if dropout and encoder.lstm_dropout else 1.0
                    cell = LayerNormBasicLSTMCell(encoder.cell_size, dropout_keep_prob=keep_prob,
                                                  layer_norm=encoder.layer_norm, reuse=reuse)
                    cell = CellWrapper(cell)
                else:
                    cell = GRUCell(encoder.cell_size, reuse=reuse)

                if dropout is not None and not (encoder.use_lstm and encoder.lstm_dropout):
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
                initial_state = get_variable(name, initializer=tf.zeros(state_size))
                return tf.tile(tf.expand_dims(initial_state, axis=0), [batch_size, 1])

            if encoder.bidir:
                rnn = lambda reuse: stack_bidirectional_dynamic_rnn(
                    cells_fw=[get_cell(reuse=reuse) for _ in range(encoder.layers)],
                    cells_bw=[get_cell(reuse=reuse) for _ in range(encoder.layers)],
                    initial_states_fw=[get_initial_state('initial_state_fw')] * encoder.layers,
                    initial_states_bw=[get_initial_state('initial_state_bw')] * encoder.layers,
                    **parameters)[0]

                try:
                    encoder_outputs_ = rnn(reuse=False)
                except ValueError:   # Multi-task scenario where we're reusing the same RNN parameters
                    encoder_outputs_ = rnn(reuse=True)

                encoder_state_ = encoder_outputs_[:, 0, encoder.cell_size:]  # first backward output
            else:
                if encoder.layers > 1:
                    cell = MultiRNNCell([get_cell() for _ in range(encoder.layers)])
                else:
                    cell = get_cell()

                initial_state = (get_initial_state(),) * encoder.layers
                encoder_outputs_, _ = auto_reuse(tf.nn.dynamic_rnn)(cell=cell, initial_state=initial_state,
                                                                    **parameters)
                encoder_state_ = encoder_outputs_[:, -1, :]

            encoder_outputs.append(encoder_outputs_)
            encoder_states.append(encoder_state_)

    encoder_state = tf.concat(encoder_states, 1)
    return encoder_outputs, encoder_state


def compute_energy(hidden, state, attn_size, **kwargs):
    input_size = hidden.get_shape()[2].value

    y = dense(state, attn_size, use_bias=True, name='W_a')
    y = tf.expand_dims(y, axis=1)

    k = get_variable('U_a', [input_size, attn_size])
    f = tf.einsum('ijk,kl->ijl', hidden, k)

    v = get_variable('v_a', [attn_size])
    s = f + y

    return tf.reduce_sum(v * tf.tanh(s), [2])


def global_attention(state, hidden_states, encoder, encoder_input_length, scope=None, context=None, **kwargs):
    with tf.variable_scope(scope or 'attention'):
        if context is not None and encoder.use_context:
            state = tf.concat([state, context], axis=1)

        e = compute_energy(hidden_states, state, attn_size=encoder.attn_size, **kwargs)
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

        if pos is None:
            wp = get_variable('Wp', [state_size, state_size])
            vp = get_variable('vp', [state_size, 1])

            pos = tf.nn.sigmoid(tf.matmul(tf.nn.tanh(tf.matmul(state, wp)), vp))
            pos = tf.floor(encoder_input_length * pos)

        pos = tf.reshape(pos, [-1, 1])
        pos = tf.minimum(pos, encoder_input_length - 1)

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


def attention_decoder(decoder_inputs, initial_state, attention_states, encoders, decoder, encoder_input_length,
                      dropout=None, feed_previous=0.0, align_encoder_id=0, encoder_inputs=None, **kwargs):
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
    # TODO: remove all "edit" stuff and put it into dual_decoder (and rename it to `edit_decoder`)
    assert decoder.cell_size % 2 == 0, 'cell size must be a multiple of 2'   # because of maxout

    embedding_shape = [decoder.vocab_size, decoder.embedding_size]

    device = None if decoder.tie_embeddings else '/cpu:0'
    with tf.device(device):
        embedding = get_variable('embedding_{}'.format(decoder.name), shape=embedding_shape)

    def embed(input_):
        return tf.nn.embedding_lookup(embedding, input_)

    def get_cell(reuse=False):
        cells = []

        for _ in range(decoder.layers):
            if decoder.use_lstm:
                keep_prob = dropout if dropout and decoder.lstm_dropout else 1.0
                cell = LayerNormBasicLSTMCell(decoder.cell_size, dropout_keep_prob=keep_prob,
                                              layer_norm=decoder.layer_norm, reuse=reuse)
                cell = CellWrapper(cell)
            else:
                cell = GRUCell(decoder.cell_size, reuse=reuse)

            if dropout is not None and not (decoder.use_lstm and decoder.lstm_dropout):
                cell = DropoutWrapper(cell, input_keep_prob=dropout)

            cells.append(cell)

        if len(cells) == 1:
            return cells[0]
        else:
            return CellWrapper(MultiRNNCell(cells))

    with tf.variable_scope('decoder_{}'.format(decoder.name)):
        attention_ = functools.partial(multi_attention, hidden_states=attention_states, encoders=encoders,
                                       encoder_input_length=encoder_input_length,
                                       aggregation_method=decoder.aggregation_method)
        input_shape = tf.shape(decoder_inputs)
        batch_size = input_shape[0]
        time_steps = input_shape[1]

        output_size = decoder.vocab_size

        state_size = get_cell().state_size

        if dropout is not None:
            initial_state = tf.nn.dropout(initial_state, dropout)

        state = dense(initial_state, state_size, use_bias=True, name='initial_state_projection', activation=tf.nn.tanh)

        if decoder.pred_state:
            pred_state = dense(initial_state, state_size, use_bias=True, name='initial_pred_state_projection',
                               activation=tf.nn.tanh)
            state = tf.concat([state, pred_state], axis=1)
            state_size *= 2

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

            output_ = tf.concat([state, context_vector], axis=1)
            # if decoder.pred_edits:
            #     output_ = tf.concat([state, context_vector], axis=1)
            # else:
            output_ = tf.concat([state, input_, context_vector], axis=1)

            if decoder.maxout:
                output_ = dense(output_, decoder.cell_size, use_bias=False, name='maxout')
                output_ = tf.reduce_max(tf.reshape(output_, tf.stack([batch_size, decoder.cell_size // 2, 2])), axis=2)

            output_ = dense(output_, decoder.embedding_size, use_bias=False, name='softmax0')
            decoder_outputs = decoder_outputs.write(time, output_)

            if decoder.tie_embeddings:
                # with tf.device('/cpu:0'):
                bias = get_variable('softmax1/bias', shape=[decoder.vocab_size])
                output_ = tf.matmul(output_, tf.transpose(embedding)) + bias
            else:
                output_ = dense(output_, output_size, use_bias=True, name='softmax1')

            proj_outputs = proj_outputs.write(time, output_)

            argmax = lambda: tf.argmax(output_, 1)
            target = lambda: inputs.read(time + 1)

            sample = tf.cond(tf.logical_and(time < time_steps - 1, tf.random_uniform([]) >= feed_previous),
                             target, argmax)
            sample.set_shape([None])
            sample = tf.stop_gradient(sample)

            def get_next_state(x, state, scope=None):
                def fun():
                    try:
                        _, new_state = get_cell()(x, state)
                    except ValueError:  # auto_reuse doesn't work with LSTM cells
                        _, new_state = get_cell(reuse=True)(x, state)
                    return new_state

                if scope is not None:
                    with tf.variable_scope(scope):
                        return fun()
                else:
                    return fun()

            if decoder.pred_edits:
                is_keep = tf.equal(sample, utils.KEEP_ID)
                is_sub = tf.equal(sample, utils.SUB_ID)
                is_del = tf.equal(sample, utils.DEL_ID)

                i = tf.logical_or(is_keep, is_sub)
                i = tf.logical_or(i, is_del)
                i = tf.to_float(i)
                edit_pos += i
                edit_pos = tf.minimum(edit_pos, tf.to_float(encoder_input_length[align_encoder_id]) - 1)
                input_ = embed(sample)

                x = tf.concat([input_, context_vector], 1)
                # x = input_
                new_state = get_next_state(x, state)
            else:
                edit_pos += 1
                input_ = embed(sample)
                #x = tf.concat([input_, context_vector], 1)
                x = input_
                new_state = get_next_state(x, state)

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



def simple_decoder(decoder_inputs, initial_state, attention_states, encoders, decoder, encoder_input_length,
                   dropout=None, feed_previous=0.0, align_encoder_id=0, **kwargs):
    device = None if decoder.tie_embeddings else '/cpu:0'
    with tf.device(device):
        embedding_shape = [decoder.vocab_size, decoder.embedding_size]
        embedding = get_variable('embedding_{}'.format(decoder.name), shape=embedding_shape)

    def embed(input_):
        return tf.nn.embedding_lookup(embedding, input_)

    def get_cell(reuse=False):
        cells = []

        for _ in range(decoder.layers):
            if decoder.use_lstm:
                keep_prob = dropout if dropout and decoder.lstm_dropout else 1.0
                cell = LayerNormBasicLSTMCell(decoder.cell_size, dropout_keep_prob=keep_prob,
                                              layer_norm=decoder.layer_norm, reuse=reuse)
                cell = CellWrapper(cell)
            else:
                cell = GRUCell(decoder.cell_size, reuse=reuse)

            if dropout is not None and not (decoder.use_lstm and decoder.lstm_dropout):
                cell = DropoutWrapper(cell, input_keep_prob=dropout)

            cells.append(cell)

        if len(cells) == 1:
            return cells[0]
        else:
            return CellWrapper(MultiRNNCell(cells))

    with tf.variable_scope('decoder_{}'.format(decoder.name)):
        attention_ = functools.partial(multi_attention, hidden_states=attention_states, encoders=encoders,
                                       encoder_input_length=encoder_input_length,
                                       aggregation_method=decoder.aggregation_method)
        input_shape = tf.shape(decoder_inputs)
        batch_size = input_shape[0]
        time_steps = input_shape[1]
        state_size = get_cell().state_size

        if dropout is not None:
            initial_state = tf.nn.dropout(initial_state, dropout)

        state = dense(initial_state, state_size, use_bias=True, name='initial_state_projection', activation=tf.nn.tanh)
        time = tf.constant(0, dtype=tf.int32, name='time')

        proj_outputs = tf.TensorArray(dtype=tf.float32, size=time_steps, clear_after_read=False)

        inputs = tf.TensorArray(dtype=tf.int64, size=time_steps, clear_after_read=False).unstack(
                                tf.cast(tf.transpose(decoder_inputs, perm=(1, 0)), tf.int64))

        def _time_step(time, state, proj_outputs):

            argmax = lambda: tf.argmax(proj_outputs.read(time - 1), axis=1)
            target = lambda: inputs.read(time)

            sample = tf.cond(tf.logical_or(tf.equal(time, 0), tf.random_uniform([]) >= feed_previous),
                             target, argmax)

            sample.set_shape([None])
            sample = tf.stop_gradient(sample)
            input_ = embed(sample)

            def get_next_state(x, state, scope=None):
                def fun():
                    try:
                        _, new_state = get_cell()(x, state)
                    except ValueError:  # auto_reuse doesn't work with LSTM cells
                        _, new_state = get_cell(reuse=True)(x, state)
                    return new_state

                if scope is not None:
                    with tf.variable_scope(scope):
                        return fun()
                else:
                    return fun()

            context_vector, new_weights = attention_(state)
            #x = tf.concat([input_, context_vector], 1)
            x = input_
            state = get_next_state(x, state)

            #output_ = state
            output_ = tf.concat([state, context_vector], axis=1)

            if decoder.maxout:
                output_ = dense(output_, decoder.cell_size, use_bias=False, name='maxout')
                output_ = tf.reduce_max(tf.reshape(output_, tf.stack([batch_size, decoder.cell_size // 2, 2])), axis=2)

            output_ = dense(output_, decoder.embedding_size, use_bias=False, name='softmax0')

            if decoder.tie_embeddings:
                bias = get_variable('softmax1/bias', shape=[decoder.vocab_size])
                output_ = tf.matmul(output_, tf.transpose(embedding)) + bias
            else:
                output_ = dense(output_, decoder.vocab_size, use_bias=True, name='softmax1')

            proj_outputs = proj_outputs.write(time, output_)

            return time + 1, state, proj_outputs

        _, new_state, proj_outputs = tf.while_loop(
            cond=lambda time, *_: time < time_steps,
            body=_time_step,
            loop_vars=(time, state, proj_outputs),
            parallel_iterations=decoder.parallel_iterations,
            swap_memory=decoder.swap_memory)

        proj_outputs = proj_outputs.stack()
        beam_tensors = utils.AttrDict(data=state, new_data=new_state)
        proj_outputs = tf.transpose(proj_outputs, perm=(1, 0, 2))

        return proj_outputs, None, None, None, None, beam_tensors



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

    return e / tf.clip_by_value(tf.reduce_sum(e, axis=dim, keep_dims=True), 10e-37, 10e+37)


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


def encoder_decoder(encoders, decoders, dropout, encoder_inputs, targets, feed_previous,
                    align_encoder_id=0, **kwargs):
    decoder = decoders[0]
    targets = targets[0]  # single decoder

    encoder_input_length = []
    for encoder_inputs_ in encoder_inputs:
        weights = get_weights(encoder_inputs_, utils.EOS_ID, time_major=False, include_first_eos=True)
        encoder_input_length.append(tf.to_int32(tf.reduce_sum(weights, axis=1)))

    parameters = dict(encoders=encoders, decoder=decoder, dropout=dropout,
                      encoder_input_length=encoder_input_length,
                      encoder_inputs=encoder_inputs)

    target_weights = get_weights(targets[:, 1:], utils.EOS_ID, time_major=False, include_first_eos=True)

    attention_states, encoder_state = multi_encoder(**parameters)

    outputs, attention_weights, _, _, _, beam_tensors = attention_decoder(   # FIXME
        attention_states=attention_states, initial_state=encoder_state,
        feed_previous=feed_previous, decoder_inputs=targets[:, :-1],
        align_encoder_id=align_encoder_id, **parameters
    )

    xent_loss = sequence_loss(logits=outputs, targets=targets[:, 1:], weights=target_weights)
    return xent_loss, [outputs], encoder_state, attention_states, attention_weights, beam_tensors


def chained_encoders(encoders, decoders, dropout, encoder_inputs, targets, feed_previous, chaining_strategy=None,
                     more_dropout=False, align_encoder_id=0, chaining_non_linearity=False, chaining_loss_ratio=1.0,
                     chaining_stop_gradient=False, **kwargs):
    decoder = decoders[0]
    targets = targets[0]  # single decoder

    assert len(encoders) == 2

    encoder_input_length = []
    input_weights = []
    for encoder_inputs_ in encoder_inputs:
        weights = get_weights(encoder_inputs_, utils.EOS_ID, time_major=False, include_first_eos=True)
        input_weights.append(weights)
        encoder_input_length.append(tf.to_int32(tf.reduce_sum(weights, axis=1)))

    target_weights = get_weights(targets[:, 1:], utils.EOS_ID, time_major=False, include_first_eos=True)

    parameters = dict(encoders=encoders[1:], decoder=encoders[0], dropout=dropout,
                      encoder_input_length=encoder_input_length[1:], more_dropout=more_dropout)

    attention_states, encoder_state = multi_encoder(encoder_inputs[1:], **parameters)

    decoder_inputs = encoder_inputs[0][:, :-1]
    batch_size = tf.shape(decoder_inputs)[0]

    pad = tf.ones(shape=tf.stack([batch_size, 1]), dtype=tf.int32) * utils.BOS_ID
    decoder_inputs = tf.concat([pad, decoder_inputs], axis=1)

    outputs, _, _, states, attns, _ = attention_decoder(
        attention_states=attention_states, initial_state=encoder_state, decoder_inputs=decoder_inputs,
        **parameters
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

    parameters = dict(encoders=encoders[:1], decoder=decoder, dropout=dropout,
                      encoder_input_length=encoder_input_length[:1],
                      encoder_inputs=encoder_inputs[:1], other_inputs=other_inputs)

    attention_states, encoder_state = multi_encoder(**parameters)

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

    outputs, attention_weights, _, _, _, beam_tensors = attention_decoder(
        attention_states=attention_states, initial_state=encoder_state,
        feed_previous=feed_previous, decoder_inputs=decoder_inputs,
        align_encoder_id=align_encoder_id, **parameters
    )

    xent_loss = sequence_loss(logits=outputs, targets=targets[:, 1:],
                                            weights=target_weights)

    if chaining_loss is not None and chaining_loss_ratio:
        xent_loss += chaining_loss_ratio * chaining_loss

    return xent_loss, [outputs], encoder_state, attention_states, attention_weights, beam_tensors


def dual_encoder_decoder(encoders, decoders, dropout, encoder_inputs, targets, feed_previous,
                         align_encoder_id=0, **kwargs):
    main_decoder = decoders[0]

    encoder_input_length = []
    for encoder_inputs_ in encoder_inputs:
        weights = get_weights(encoder_inputs_, utils.EOS_ID, time_major=False, include_first_eos=True)
        encoder_input_length.append(tf.to_int32(tf.reduce_sum(weights, axis=1)))

    parameters = dict(encoders=encoders, decoders=decoders, dropout=dropout,
                      encoder_input_length=encoder_input_length,
                      encoder_inputs=encoder_inputs)

    target_weights = [
        get_weights(targets_[:, 1:], utils.EOS_ID, time_major=False, include_first_eos=True)
        for targets_ in targets
    ]

    if main_decoder.ignore_dels:
        # Ignore <DEL> symbols, in the word output (those are used as dummy words for the <DEL> op)
        weights = target_weights[-1]
        target_weights[-1] = tf.to_float(tf.not_equal(weights, utils.DEL_ID)) * weights

    decoder_inputs = [targets_[:, :-1] for targets_ in targets]

    attention_states, encoder_state = multi_encoder(**parameters)

    outputs, attention_weights, _, _, beam_tensors = dual_decoder(
        attention_states=attention_states, initial_state=encoder_state,
        feed_previous=feed_previous, decoder_inputs=decoder_inputs,
        align_encoder_id=align_encoder_id, **parameters
    )

    ratios = [main_decoder.get('op_loss_ratio', 1.0), 1.0]
    ratios = [ratio / sum(ratios) for ratio in ratios]

    xent_loss = [
        sequence_loss(logits=outputs_, targets=targets_[:, 1:], weights=weights_)
        for outputs_, targets_, weights_ in zip(outputs, targets, target_weights)
    ]

    if main_decoder.op_loss_strategy == 'random':
        xent_loss = tf.cond(
            tf.random_uniform([]) < ratios[0],
            lambda: xent_loss[0],
            lambda: xent_loss[1]
        )
    else:
        xent_loss = sum([ratio * loss_ for ratio, loss_ in zip(ratios, xent_loss)])

    return xent_loss, outputs, encoder_state, attention_states, attention_weights, beam_tensors


def dual_decoder(decoder_inputs, initial_state, attention_states, encoders, decoders, encoder_input_length,
                 dropout=None, feed_previous=0.0, align_encoder_id=0, encoder_inputs=None, **kwargs):
    main_decoder = decoders[0]   # main decoder is the first one
    assert main_decoder.cell_size % 2 == 0, 'cell size must be a multiple of 2'   # because of maxout

    embeddings = []
    for decoder_ in decoders:
        device = None if decoder_.tie_embeddings else '/cpu:0'
        with tf.device(device):
            embedding = get_variable('embedding_{}'.format(decoder_.name),
                                     shape=[decoder_.vocab_size, decoder_.embedding_size])
        embeddings.append(embedding)

    def embed(input_, encoder_id):
        return tf.nn.embedding_lookup(embeddings[encoder_id], input_)

    def get_cell(reuse=False):
        cells = []

        for _ in range(main_decoder.layers):
            if main_decoder.use_lstm:
                keep_prob = dropout if dropout and main_decoder.lstm_dropout else 1.0
                cell = LayerNormBasicLSTMCell(main_decoder.cell_size, dropout_keep_prob=keep_prob,
                                              layer_norm=main_decoder.layer_norm, reuse=reuse)
                cell = CellWrapper(cell)
            else:
                cell = GRUCell(main_decoder.cell_size, reuse=reuse)

            if dropout is not None and not (main_decoder.use_lstm and main_decoder.lstm_dropout):
                cell = DropoutWrapper(cell, input_keep_prob=dropout)

            cells.append(cell)

        if len(cells) == 1:
            return cells[0]
        else:
            return CellWrapper(MultiRNNCell(cells))

    attention_ = functools.partial(multi_attention, hidden_states=attention_states, encoders=encoders,
                                   encoder_input_length=encoder_input_length,
                                   aggregation_method=main_decoder.aggregation_method)

    input_shape = tf.shape(decoder_inputs[0])
    batch_size = input_shape[0]
    time_steps = input_shape[1]
    state_size = get_cell().state_size

    if dropout is not None:
        initial_state = tf.nn.dropout(initial_state, dropout)

    with tf.variable_scope('decoder_{}'.format(main_decoder.name)):
        state = dense(initial_state, state_size, use_bias=True, name='initial_state_projection', activation=tf.nn.tanh)

    edit_pos = tf.zeros([batch_size], tf.float32)

    # used by beam-search decoder (by dict feeding)
    data = tf.concat([state, tf.expand_dims(edit_pos, axis=1)], axis=1)
    state, edit_pos = tf.split(data, [state_size, 1], axis=1)
    edit_pos = tf.squeeze(edit_pos, axis=1)

    time = tf.constant(0, dtype=tf.int32, name='time')
    proj_outputs = tf.TensorArray(dtype=tf.float32, size=time_steps, clear_after_read=False)

    inputs = [
        tf.TensorArray(dtype=tf.int64, size=time_steps, clear_after_read=False).unstack(
            tf.cast(tf.transpose(decoder_inputs[i], perm=(1, 0)), tf.int64))
        for i in range(len(decoders))
    ]

    states = tf.TensorArray(dtype=tf.float32, size=time_steps)
    weights = tf.TensorArray(dtype=tf.float32, size=time_steps)
    attns = tf.TensorArray(dtype=tf.float32, size=time_steps)

    def aggregate_inputs(inputs_, method='concat'):
        if method == 'only_ops':
            return inputs_[0]
        elif method == 'only_words':
            return inputs_[1]
        elif method == 'sum':
            return sum(inputs_)
        else:
            return tf.concat(inputs_, axis=1)

    def _time_step(time, state, proj_outputs, states, weights, edit_pos, attns):
        pos = [edit_pos if encoder.align_edits else None for encoder in encoders]

        with tf.variable_scope('decoder_{}'.format(main_decoder.name)):
            context_vector, new_weights = attention_(state, pos=pos)
            attns = attns.write(time, context_vector)
            weights = weights.write(time, new_weights[align_encoder_id])

            output_ = tf.concat([state, context_vector], axis=1)
            output_ = dense(output_, main_decoder.cell_size, use_bias=False, name='maxout')
            output_ = tf.reduce_max(tf.reshape(output_, tf.stack([batch_size, main_decoder.cell_size // 2, 2])), axis=2)
            #output_ = dense(output_, main_decoder.embedding_size, use_bias=False, name='softmax0')

        outputs_ = []
        for i, decoder_ in enumerate(decoders):
            with tf.variable_scope('decoder_{}'.format(decoder_.name)):
                new_output_ = dense(output_, decoder_.embedding_size, use_bias=False, name='softmax0')

                if decoder_.tie_embeddings:
                    bias = get_variable('softmax1_{}/bias'.format(decoder_.name), shape=[decoder_.vocab_size])
                    new_output_ = tf.matmul(new_output_, tf.transpose(embeddings[i])) + bias
                else:
                    new_output_ = dense(new_output_, decoder_.vocab_size, use_bias=True, name='softmax1')

                outputs_.append(new_output_)

        output_ = tf.concat(outputs_, axis=1)
        proj_outputs = proj_outputs.write(time, output_)

        samples = []  # generated symbols: groundtruth (teacher forcing) when training,
        # or argmax when testing

        for inputs_, output_ in zip(inputs, outputs_):
            argmax = lambda: tf.argmax(output_, 1)
            target = lambda: inputs_.read(time + 1)
            sample = tf.cond(tf.logical_and(time < time_steps - 1, tf.random_uniform([]) >= feed_previous),
                             target, argmax)
            sample.set_shape([None])
            sample = tf.stop_gradient(sample)
            samples.append(sample)

        sample = samples[0]
        is_keep = tf.equal(sample, utils.KEEP_ID)

        if main_decoder.force_keep:  # when predicted op is KEEP, force the decoder to output
            # the aligned input word.
            # For this to work, the embeddings must be shared between input and output (same
            # embedding names)
            aligned_word = tf.gather_nd(encoder_inputs[align_encoder_id],
                                        tf.stack([tf.range(batch_size), tf.to_int32(edit_pos)], axis=1))
            samples[1] = tf.where(
                is_keep,
                tf.to_int64(aligned_word),
                samples[1]
            )

        is_sub = tf.equal(sample, utils.SUB_ID)
        is_del = tf.equal(sample, utils.DEL_ID)
        i = tf.logical_or(is_keep, is_sub)
        i = tf.logical_or(i, is_del)
        i = tf.to_float(i)
        edit_pos += i
        edit_pos = tf.minimum(edit_pos, tf.to_float(encoder_input_length[align_encoder_id]) - 1)

        embedded_inputs = [embed(sample_, i) for i, sample_ in enumerate(samples)]

        output_aggregation = main_decoder.output_aggregation or main_decoder.input_aggregation
        input_ = aggregate_inputs(embedded_inputs, method=output_aggregation)

        with tf.variable_scope('decoder_{}'.format(main_decoder.name)):
            try:
                _, new_state = get_cell()(input_, state)
            except ValueError:  # auto_reuse doesn't work with LSTM cells
                _, new_state = get_cell(reuse=True)(input_, state)

        if main_decoder.skip_update:  # when generated op is DEL, the generated word
            # isn't useful to the language model, so we don't update the LSTM's state
            new_state = tf.where(
                is_del,
                state,
                new_state
            )

        states = states.write(time, new_state)

        return time + 1, new_state, proj_outputs, states, weights, edit_pos, attns

    _, new_state, proj_outputs, states, weights, new_edit_pos, attns = tf.while_loop(
        cond=lambda time, *_: time < time_steps,
        body=_time_step,
        loop_vars=(time, state, proj_outputs, weights, states, edit_pos, attns),
        parallel_iterations=main_decoder.parallel_iterations,
        swap_memory=main_decoder.swap_memory)

    proj_outputs = proj_outputs.stack()
    weights = weights.stack()  # batch_size, encoders, output time, input time
    states = states.stack()
    attns = attns.stack()

    new_data = tf.concat([new_state, tf.expand_dims(new_edit_pos, axis=1)], axis=1)

    beam_tensors = utils.AttrDict(data=data, new_data=new_data)

    proj_outputs = tf.transpose(proj_outputs, perm=(1, 0, 2))

    weights = tf.transpose(weights, perm=(1, 0, 2))
    states = tf.transpose(states, perm=(1, 0, 2))
    attns = tf.transpose(attns, perm=(1, 0, 2))

    proj_outputs = tf.split(proj_outputs, [decoder.vocab_size for decoder in decoders], axis=2)

    return proj_outputs, weights, states, attns, beam_tensors
