import numpy as np
import tensorflow as tf
import re
import itertools

from translate import utils
from translate import decoders
from collections import namedtuple


class Seq2SeqModel(object):
    def __init__(self, encoders, decoder, learning_rate, global_step, max_gradient_norm, dropout_rate=0.0,
                 freeze_variables=None, max_output_len=50, feed_previous=0.0,
                 optimizer='sgd', max_input_len=None, decode_only=False, len_normalization=1.0,
                 chained_encoders=False, chaining_strategy=None, more_dropout=False, align_encoder_id=0,
                 **kwargs):
        self.encoders = encoders
        self.decoder = decoder

        self.learning_rate = learning_rate
        self.global_step = global_step

        self.encoder_count = len(encoders)
        self.trg_vocab_size = decoder.vocab_size
        self.trg_cell_size = decoder.cell_size

        self.max_output_len = max_output_len
        self.max_input_len = max_input_len
        self.len_normalization = len_normalization

        if dropout_rate > 0:
            self.dropout = tf.Variable(1 - dropout_rate, trainable=False, name='dropout_keep_prob')
            self.dropout_off = self.dropout.assign(1.0)
            self.dropout_on = self.dropout.assign(1 - dropout_rate)
        else:
            self.dropout = None

        self.feed_previous = tf.constant(feed_previous, dtype=tf.float32)
        self.feed_argmax = tf.constant(True, dtype=tf.bool)  # feed with argmax or sample

        self.encoder_inputs = []
        self.encoder_input_length = []
        self.input_weights = []

        self.extensions = [encoder.name for encoder in encoders] + [decoder.name]
        self.encoder_names = [encoder.name for encoder in encoders]
        self.decoder_name = decoder.name
        self.extensions = self.encoder_names + [self.decoder_name]
        self.freeze_variables = freeze_variables or []
        self.max_gradient_norm = max_gradient_norm

        for encoder in self.encoders:
            placeholder = tf.placeholder(tf.int32, shape=[None, None], name='encoder_{}'.format(encoder.name))

            self.encoder_inputs.append(placeholder)
            weights = decoders.get_weights(placeholder, utils.EOS_ID, time_major=False, include_first_eos=True)
            self.input_weights.append(weights)
            self.encoder_input_length.append(tf.to_int32(tf.reduce_sum(weights, axis=1)))

        # starts with BOS, and ends with EOS
        self.targets = tf.placeholder(tf.int32, shape=[None, None], name='target_{}'.format(self.decoder.name))
        self.decoder_inputs = self.targets[:,:-1]

        self.target_weights = decoders.get_weights(self.targets[:,1:], utils.EOS_ID, time_major=False,
                                                   include_first_eos=True)

        self.losses = []

        if chained_encoders:
            assert len(encoders) == 2

            parameters = dict(encoders=encoders[1:], decoder=encoders[0], dropout=self.dropout,
                              encoder_input_length=self.encoder_input_length[1:], more_dropout=more_dropout)

            attention_states, encoder_state = decoders.multi_encoder(self.encoder_inputs[1:], **parameters)

            decoder_inputs = self.encoder_inputs[0][:,:-1]
            batch_size = tf.shape(decoder_inputs)[0]

            pad = tf.ones(shape=tf.stack([batch_size, 1]), dtype=tf.int32) * utils.BOS_ID
            decoder_inputs = tf.concat([pad, decoder_inputs], axis=1)

            outputs, _, _, states, attns, _ = decoders.attention_decoder(
                attention_states=attention_states, initial_state=encoder_state, decoder_inputs=decoder_inputs,
                **parameters
            )

            chaining_loss = decoders.sequence_loss(logits=outputs, targets=self.encoder_inputs[0],
                                                   weights=self.input_weights[0])
            self.losses.append(chaining_loss)

            if decoder.use_lstm:
                size = states.get_shape()[2].value
                decoder_outputs = states[:,:,size // 2:]
            else:
                decoder_outputs = states

            if chaining_strategy == 'share_states':
                other_inputs = states
            elif chaining_strategy == 'share_outputs':
                other_inputs = decoder_outputs
            else:
                other_inputs = None

            if other_inputs is not None and kwargs.get('chaining_stop_gradient'):
                other_inputs = tf.stop_gradient(other_inputs)

            parameters = dict(encoders=encoders[:1], decoder=decoder, dropout=self.dropout,
                              encoder_input_length=self.encoder_input_length[:1],
                              encoder_inputs=self.encoder_inputs[:1], other_inputs=other_inputs)
        else:
            chaining_loss = None
            parameters = dict(encoders=encoders, decoder=decoder, dropout=self.dropout,
                              encoder_input_length=self.encoder_input_length,
                              encoder_inputs=self.encoder_inputs)
            states = None
            attns = None
            decoder_outputs = None

        self.attention_states, self.encoder_state = decoders.multi_encoder(**parameters)

        if chained_encoders:
            if self.dropout is not None and more_dropout:
                attns = tf.nn.dropout(attns, keep_prob=self.dropout)
                states = tf.nn.dropout(states, keep_prob=self.dropout)
                decoder_outputs = tf.nn.dropout(decoder_outputs, keep_prob=self.dropout)

            if kwargs.get('chaining_stop_gradient'):
                attns = tf.stop_gradient(attns)
                states = tf.stop_gradient(states)
                decoder_outputs = tf.stop_gradient(decoder_outputs)

            if chaining_strategy == 'concat_attns':
                self.attention_states[0] = tf.concat([self.attention_states[0], attns], axis=2)
            elif chaining_strategy == 'concat_states':
                self.attention_states[0] = tf.concat([self.attention_states[0], states], axis=2)
            elif chaining_strategy == 'sum_attns':
                self.attention_states[0] += attns
            elif chaining_strategy in ('map_attns', 'map_states', 'map_outputs'):
                if chaining_strategy == 'map_attns':
                    x = attns
                elif chaining_strategy == 'map_outputs':
                    x = decoder_outputs
                else:
                    x = states

                shape = [x.get_shape()[-1], self.attention_states[0].get_shape()[-1]]

                w = tf.get_variable("map_attns/Matrix", shape=shape)
                b = tf.get_variable("map_attns/bias", shape=shape[-1:])

                x = tf.einsum('ijk,kl->ijl', x, w) + b
                if kwargs.get('chaining_non_linearity'):
                    x = tf.nn.tanh(x)
                    if self.dropout is not None and more_dropout:
                        x = tf.nn.dropout(x, keep_prob=self.dropout)

                self.attention_states[0] += x

        self.outputs, self.attention_weights, _, _, _, self.beam_tensors = decoders.attention_decoder(
            attention_states=self.attention_states, initial_state=self.encoder_state,
            feed_previous=self.feed_previous, decoder_inputs=self.decoder_inputs,
            align_encoder_id=align_encoder_id, **parameters
        )
        self.beam_output = decoders.softmax(self.outputs[:, 0, :])

        self.xent_loss = decoders.sequence_loss(logits=self.outputs, targets=self.targets[:, 1:],
                                                weights=self.target_weights)
        self.losses.append(self.xent_loss)

        chaining_loss_ratio = kwargs.get('chaining_loss_ratio')
        if chaining_loss is not None and chaining_loss_ratio:
            self.xent_loss += chaining_loss_ratio * chaining_loss

        optimizers = self.get_optimizers(optimizer, learning_rate)
        if not decode_only:
            self.update_op, self.sgd_update_op = self.get_update_op(self.xent_loss, optimizers, self.global_step)

    @staticmethod
    def get_optimizers(optimizer_name, learning_rate):
        sgd_opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        if optimizer_name.lower() == 'adadelta':
            # same epsilon and rho as Bahdanau et al. 2015
            opt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, epsilon=1e-06, rho=0.95)
        elif optimizer_name.lower() == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        else:
            opt = sgd_opt

        return opt, sgd_opt

    def get_update_op(self, loss, opts, global_step=None):
        # compute gradient only for variables that are not frozen
        frozen_parameters = [var.name for var in tf.trainable_variables()
                             if any(re.match(var_, var.name) for var_ in self.freeze_variables)]
        params = [var for var in tf.trainable_variables() if var.name not in frozen_parameters]

        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

        update_ops = []
        for opt in opts:
            update_op = opt.apply_gradients(zip(clipped_gradients, params), global_step=global_step)
            update_ops.append(update_op)

        return update_ops

    def step(self, session, data, update_model=True, align=False, use_sgd=False, **kwargs):
        if self.dropout is not None:
            session.run(self.dropout_on)

        encoder_inputs, targets = self.get_batch(data)

        input_feed = {self.targets: targets}

        for i in range(self.encoder_count):
            input_feed[self.encoder_inputs[i]] = encoder_inputs[i]

        output_feed = {'loss': self.losses[-1]}
        if update_model:
            output_feed['updates'] = self.sgd_update_op if use_sgd else self.update_op
        if align:
            output_feed['weights'] = self.attention_weights

        res = session.run(output_feed, input_feed)

        return namedtuple('output', 'loss weights')(res['loss'], res.get('weights'))

    def greedy_decoding(self, session, token_ids):
        if self.dropout is not None:
            session.run(self.dropout_off)

        data = [
            ids + [[]] if len(ids) == self.encoder_count else ids
            for ids in token_ids
        ]

        batch = self.get_batch(data, decoding=True)
        encoder_inputs, targets = batch

        input_feed = {self.targets: targets, self.feed_previous: 1.0}

        for i in range(self.encoder_count):
            input_feed[self.encoder_inputs[i]] = encoder_inputs[i]

        outputs = session.run(self.outputs, input_feed)
        return np.argmax(outputs, axis=2)

    def beam_search_decoding(self, session, token_ids, beam_size, early_stopping=True, vocabs=None):
        if not isinstance(session, list):
            session = [session]

        if self.dropout is not None:
            for session_ in session:
                session_.run(self.dropout_off)

        data = [token_ids + [[]]]
        encoder_inputs, targets = self.get_batch(data, decoding=True)
        input_feed = {}

        for i in range(self.encoder_count):
            input_feed[self.encoder_inputs[i]] = encoder_inputs[i]

        output_feed = [self.encoder_state] + self.attention_states
        res = [session_.run(output_feed, input_feed) for session_ in session]
        state, attn_states = list(zip(*[(res_[0], res_[1:]) for res_ in res]))

        targets = targets[:,0]  # BOS symbol

        finished_hypotheses = []
        finished_scores = []

        hypotheses = [[]]
        scores = np.zeros([1], dtype=np.float32)

        beam_data = None

        for k in range(self.max_output_len):
            batch_size = targets.shape[0]
            targets = np.reshape(targets, [batch_size, 1])
            targets = np.concatenate([targets, np.ones(targets.shape) * utils.EOS_ID], axis=1)

            input_feed = [{self.targets: targets} for _ in session]

            if beam_data is not None:
                for feed, data_ in zip(input_feed, beam_data):
                    feed[self.beam_tensors.data] = data_

            for feed, attn_states_ in zip(input_feed, attn_states):
                for i in range(self.encoder_count):
                    feed[self.encoder_inputs[i]] = encoder_inputs[i]
                    feed[self.attention_states[i]] = attn_states_[i].repeat(batch_size, axis=0)

            output_feed = [self.beam_tensors.new_data, self.beam_output]

            res = [session_.run(output_feed, input_feed_) for session_, input_feed_ in zip(session, input_feed)]
            beam_data, proba = list(zip(*res))

            proba = [np.maximum(proba_, 1e-10) for proba_ in proba]
            scores_ = scores[:, None] - np.average([np.log(proba_) for proba_ in proba], axis=0)
            scores_ = scores_.flatten()
            flat_ids = np.argsort(scores_)

            token_ids_ = flat_ids % self.trg_vocab_size
            hyp_ids = flat_ids // self.trg_vocab_size

            new_hypotheses = []
            new_scores = []
            new_data = [[] for _ in session]
            new_input = []
            new_beam_size = beam_size

            greedy_tokens = {}

            for flat_id, hyp_id, token_id in zip(flat_ids, hyp_ids, token_ids_):
                hypothesis = hypotheses[hyp_id] + [token_id]
                score = scores_[flat_id]

                if self.decoder.pred_edits:
                    # always follow greedy recommendation
                    op_id = min(token_id, len(utils._START_VOCAB))
                    greedy_tokens.setdefault(hyp_id, op_id)

                if token_id == utils.EOS_ID:
                    # hypothesis is finished, it is thus unnecessary to keep expanding it
                    finished_hypotheses.append(hypothesis)
                    finished_scores.append(score)

                    # early stop: number of possible hypotheses is reduced by one
                    if early_stopping:
                        new_beam_size -= 1
                else:
                    new_hypotheses.append(hypothesis)

                    for session_id, data_, in enumerate(beam_data):
                        new_data[session_id].append(data_[hyp_id])

                    new_scores.append(score)
                    new_input.append(token_id)

                if len(new_hypotheses) == beam_size:
                    break

            beam_size = new_beam_size
            hypotheses = new_hypotheses
            beam_data = [np.array(data_) for data_ in new_data]
            scores = np.array(new_scores)
            targets = np.array(new_input, dtype=np.int32)

            if beam_size <= 0 or len(hypotheses) == 0:
                break

        hypotheses += finished_hypotheses
        scores = np.concatenate([scores, finished_scores])

        if self.len_normalization > 0:  # normalize score by length (to encourage longer sentences)
            scores /= [len(hypothesis) ** self.len_normalization for hypothesis in hypotheses]

        # sort best-list by score
        sorted_idx = np.argsort(scores)
        hypotheses = np.array(hypotheses)[sorted_idx].tolist()
        scores = scores[sorted_idx].tolist()
        return hypotheses, scores

    def get_batch(self, data, decoding=False):
        """
        :param data:
        :param decoding: set this parameter to True to output dummy
          data for the decoder side (using the maximum output size)
        :return:
        """
        inputs = [[] for _ in range(self.encoder_count)]
        targets = []

        # maximum input length of each encoder in this batch
        max_input_len = [max(len(data_[i]) for data_ in data) for i in range(self.encoder_count)]
        if self.max_input_len is not None:
            max_input_len = [min(len_, self.max_input_len) for len_ in max_input_len]

        # maximum output length in this batch
        max_output_len = min(max(len(data_[-1]) for data_ in data), self.max_output_len)

        for *src_sentences, trg_sentence in data:
            for i, (encoder, src_sentence) in enumerate(zip(self.encoders, src_sentences)):
                # pad sequences so that all sequences in the same batch have the same length
                src_sentence = src_sentence[:max_input_len[i]]
                encoder_pad = [utils.EOS_ID] * (1 + max_input_len[i] - len(src_sentence))
                inputs[i].append(src_sentence + encoder_pad)

            trg_sentence = trg_sentence[:max_output_len]
            if decoding:
                targets.append([utils.BOS_ID] * self.max_output_len + [utils.EOS_ID])
            else:
                decoder_pad_size = max_output_len - len(trg_sentence) + 1
                trg_sentence = [utils.BOS_ID] + trg_sentence + [utils.EOS_ID] * decoder_pad_size
                targets.append(trg_sentence)

        # convert lists to numpy arrays
        inputs = [np.array(inputs_, dtype=np.int32) for ext, inputs_ in zip(self.encoder_names, inputs)]
        targets = np.array(targets) # starts with BOS and ends with EOS

        return inputs, targets
