# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sequence-to-sequence model with an attention mechanism."""

import numpy as np
import tensorflow as tf
import re
import itertools

from translate import utils, evaluation
from translate import decoders
from collections import namedtuple


class Seq2SeqModel(object):
    """Sequence-to-sequence model with attention.

    This class implements a multi-layer recurrent neural network as encoder,
    and an attention-based decoder. This is the same as the model described in
    this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
    or into the seq2seq library for complete model implementation.
    This class also allows to use GRU cells in addition to LSTM cells, and
    sampled softmax to handle large output vocabulary size. A single-layer
    version of this model, but with bi-directional encoder, was presented in
      http://arxiv.org/abs/1409.0473
    and sampled softmax is described in Section 3 of the following paper.
      http://arxiv.org/abs/1412.2007
    """

    def __init__(self, encoders, decoder, learning_rate, global_step, max_gradient_norm, dropout_rate=0.0,
                 freeze_variables=None, max_output_len=50, feed_previous=0.0,
                 optimizer='sgd', max_input_len=None, decode_only=False, len_normalization=1.0,
                 chained_encoders=False, **kwargs):
        self.encoders = encoders
        self.decoder = decoder
        self.pred_edits = decoder.pred_edits

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

        self.extensions = [encoder.name for encoder in encoders] + [decoder.name]
        self.encoder_names = [encoder.name for encoder in encoders]
        self.decoder_name = decoder.name
        self.extensions = self.encoder_names + [self.decoder_name]
        self.freeze_variables = freeze_variables or []
        self.max_gradient_norm = max_gradient_norm

        for encoder in self.encoders:
            # batch_size x time
            placeholder = tf.placeholder(tf.int32, shape=[None, None],
                                         name='encoder_{}'.format(encoder.name))

            self.encoder_inputs.append(placeholder)
            self.encoder_input_length.append(
                tf.placeholder(tf.int64, shape=[None], name='encoder_{}_length'.format(encoder.name))
            )


        # starts with BOS, and ends with EOS  (time x batch_size)
        self.targets = tf.placeholder(tf.int32, shape=[None, None], name='target_{}'.format(self.decoder.name))
        self.decoder_inputs = self.targets[:-1,:]

        self.target_weights = decoders.get_weights(self.targets[1:,:], utils.EOS_ID, time_major=True,
                                                   include_first_eos=True)
        self.target_length = tf.reduce_sum(self.target_weights, axis=0)

        if chained_encoders:
            parameters = dict(encoders=encoders[1:], decoder=encoders[0], dropout=self.dropout,
                              encoder_input_length=self.encoder_input_length[1:])

            # (batch_size, time_steps)
            attention_states, encoder_state = decoders.multi_encoder(self.encoder_inputs[1:], **parameters)

            # import ipdb; ipdb.set_trace()
            # FIXME decoder_inputs/encoder_inputs
            # states: (time_steps, batch_size)
            # encoder_inputs: (batch_size, time_steps)

            decoder_inputs = tf.transpose(self.encoder_inputs[0], perm=(1, 0))
            batch_size = tf.shape(decoder_inputs)[1]
            pad = tf.ones(shape=tf.stack([1, batch_size]), dtype=tf.int32) * utils.BOS_ID
            decoder_inputs = tf.concat([pad, decoder_inputs], axis=0)
            decoder_input_length = 1 + self.encoder_input_length[0]

            outputs, attention_weights, decoder_outputs, sampled_output, states = decoders.attention_decoder(
                attention_states=attention_states, initial_state=encoder_state,
                decoder_inputs=decoder_inputs, decoder_input_length=decoder_input_length,
                **parameters
            )

            states = tf.transpose(states, perm=(1, 0, 2))
            self.attention_states = [states]   # or decoder_outputs
            self.encoder_state = encoder_state

            parameters = dict(encoders=encoders[:1], decoder=decoder, dropout=self.dropout,
                              encoder_input_length=self.encoder_input_length[:1])

            # import ipdb; ipdb.set_trace()
        else:
            parameters = dict(encoders=encoders, decoder=decoder, dropout=self.dropout,
                              encoder_input_length=self.encoder_input_length)
            self.attention_states, self.encoder_state = decoders.multi_encoder(self.encoder_inputs, **parameters)

        # self.attention_states[0] = tf.Print(self.attention_states[0], [tf.shape(states)], summarize=1000)

        (self.outputs, self.attention_weights, self.decoder_outputs, self.sampled_output,
         self.states) = decoders.attention_decoder(
            attention_states=self.attention_states, initial_state=self.encoder_state,
            feed_previous=self.feed_previous, decoder_inputs=self.decoder_inputs,
            decoder_input_length=self.target_length, feed_argmax=self.feed_argmax, **parameters
        )

        # self.beam_tensors = decoders.beam_attention_decoder(
        #     attention_states=self.attention_states, initial_state=self.encoder_state,
        #     feed_previous=self.feed_previous, decoder_inputs=self.decoder_inputs,
        #     decoder_input_length=self.target_length, feed_argmax=self.feed_argmax, **parameters
        # )
        #
        # self.beam_output = decoders.softmax(self.outputs[0, :, :], temperature=softmax_temperature)

        optimizers = self.get_optimizers(optimizer, learning_rate)

        self.xent_loss = None
        self.update_op, self.sgd_update_op = None, None

        self.init_xent(optimizers, decode_only)

    @staticmethod
    def get_optimizers(optimizer_name, learning_rate):
        sgd_opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        if optimizer_name.lower() == 'adadelta':
            # same epsilon and rho as Bahdanau et al. 2015
            opt = tf.train.AdadeltaOptimizer(learning_rate=1.0, epsilon=1e-06, rho=0.95)
        elif optimizer_name.lower() == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=0.001)
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

    def init_xent(self, optimizers, decode_only=False):
        self.xent_loss = decoders.sequence_loss(logits=self.outputs, targets=self.targets[1:, :],  # skip BOS
                                                weights=self.target_weights)

        if not decode_only:
            self.update_op, self.sgd_update_op = self.get_update_op(self.xent_loss, optimizers, self.global_step)

    def step(self, session, data, update_model=True, align=False, use_sgd=False, **kwargs):
        if self.dropout is not None:
            session.run(self.dropout_on)

        batch = self.get_batch(data)
        encoder_inputs, targets, encoder_input_length = batch

        input_feed = {}
        input_feed[self.targets] = targets

        for i in range(self.encoder_count):
            input_feed[self.encoder_input_length[i]] = encoder_input_length[i]
            input_feed[self.encoder_inputs[i]] = encoder_inputs[i]

        output_feed = {'loss': self.xent_loss}
        if update_model:
            output_feed['updates'] = self.sgd_update_op if use_sgd else self.update_op
        if align:
            output_feed['attn_weights'] = self.attention_weights

        res = session.run(output_feed, input_feed)

        return namedtuple('output', 'loss attn_weights')(res['loss'], res.get('attn_weights'))


    def greedy_decoding(self, session, token_ids):
        if self.dropout is not None:
            session.run(self.dropout_off)

        data = [
            ids + [[]] if len(ids) == self.encoder_count else ids
            for ids in token_ids
        ]

        batch = self.get_batch(data, decoding=True)
        encoder_inputs, targets, encoder_input_length = batch

        input_feed = {self.targets: targets, self.feed_previous: 1.0}

        for i in range(self.encoder_count):
            input_feed[self.encoder_input_length[i]] = encoder_input_length[i]
            input_feed[self.encoder_inputs[i]] = encoder_inputs[i]

        outputs = session.run(self.outputs, input_feed)

        return np.argmax(outputs, axis=2).T

    def beam_search_decoding(self, session, token_ids, beam_size, early_stopping=True, *args, **kwargs):
        if self.dropout is not None:
            session.run(self.dropout_off)

        data = [token_ids + [[]]]

        batch = self.get_batch(data, decoding=False)
        encoder_inputs, _, encoder_input_length = batch
        input_feed = {}

        for i in range(self.encoder_count):
            input_feed[self.encoder_input_length[i]] = encoder_input_length[i]
            input_feed[self.encoder_inputs[i]] = encoder_inputs[i]

        output_feed = [self.encoder_state, self.attention_states, self.beam_tensors.initial_data,
                       self.beam_tensors.initial_output]

        initial_state, attn_states, data, output = session.run(output_feed, input_feed)

        hypotheses = [[] * beam_size]
        scores = np.zeros([1], dtype=np.float32)

        finished_hypotheses = []
        finished_scores = []

        for i in range(self.max_output_len):
            if beam_size <= 0:
                break

            output = np.maximum(output, 1e-12)   # avoid division by zero
            scores_ = scores[:, None] - np.log(output)
            scores_ = scores_.flatten()
            flat_ids = np.argsort(scores_)
            token_ids_ = flat_ids % self.trg_vocab_size
            hyp_ids = flat_ids // self.trg_vocab_size

            new_hypotheses = []
            new_scores = []
            new_data = []
            new_target = []
            new_beam_size = beam_size

            for flat_id, hyp_id, token_id in zip(flat_ids, hyp_ids, token_ids_):
                hypothesis = hypotheses[hyp_id] + [token_id]
                score = scores_[flat_id]

                if token_id == utils.EOS_ID:
                    # hypothesis is finished, it is thus unnecessary to keep expanding it
                    finished_hypotheses.append(hypothesis)
                    finished_scores.append(score)

                    # early stop: number of possible hypotheses is reduced by one
                    if early_stopping:
                        new_beam_size -= 1
                else:
                    new_hypotheses.append(hypothesis)
                    new_scores.append(score)
                    new_data.append(data[hyp_id])
                    new_target.append(token_id)

                if len(new_hypotheses) == beam_size:
                    break

            beam_size = new_beam_size

            hypotheses = new_hypotheses
            scores = np.array(new_scores)
            targets = np.array(new_target, dtype=np.int32)
            data = np.array(new_data)

            batch_size = data.shape[0]
            input_feed = {
                self.beam_tensors.data: data,
                self.beam_tensors.decoder_input: targets,
                self.encoder_state: initial_state.repeat(batch_size, axis=0)
            }

            for j in range(self.encoder_count):
                input_feed[self.attention_states[j]] = attn_states[j].repeat(batch_size, axis=0)
                input_feed[self.encoder_input_length[j]] = encoder_input_length[j]

            output_feed = [self.beam_tensors.new_data,
                           self.beam_tensors.output]

            data, output = session.run(output_feed, input_feed)

        hypotheses += finished_hypotheses
        scores = np.concatenate([scores, finished_scores])

        if self.pred_edits:
            hypothesis_len = [len(hyp) - hyp.count(utils.DEL_ID) for hyp in hypotheses]
        else:
            hypothesis_len = map(len, hypotheses)

        if self.len_normalization > 0:  # normalize score by length (to encourage longer sentences)
            scores /= [len_ ** self.len_normalization for len_ in hypothesis_len]

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
        input_length = [[] for _ in range(self.encoder_count)]
        targets = []

        # maximum input length of each encoder in this batch
        max_input_len = [max(len(data_[i]) for data_ in data) for i in range(self.encoder_count)]
        if self.max_input_len is not None:
            max_input_len = [min(len_, self.max_input_len) for len_ in max_input_len]

        # maximum output length in this batch
        max_output_len = min(max(len(data_[-1]) for data_ in data), self.max_output_len)

        for *src_sentences, trg_sentence in data:
            for i, (encoder, src_sentence) in enumerate(zip(self.encoders, src_sentences)):
                pad = utils.EOS_ID

                # pad sequences so that all sequences in the same batch have the same length
                src_sentence = src_sentence[:max_input_len[i]]
                encoder_pad = [pad] * (1 + max_input_len[i] - len(src_sentence))

                inputs[i].append(src_sentence + encoder_pad)
                input_length[i].append(len(src_sentence) + 1)

            trg_sentence = trg_sentence[:max_output_len]
            if decoding:
                targets.append([utils.BOS_ID] * self.max_output_len + [utils.EOS_ID])
            else:
                decoder_pad_size = max_output_len - len(trg_sentence) + 1
                trg_sentence = [utils.BOS_ID] + trg_sentence + [utils.EOS_ID] * decoder_pad_size
                targets.append(trg_sentence)

        # convert lists to numpy arrays
        input_length = [np.array(input_length_, dtype=np.int32) for input_length_ in input_length]
        inputs = [
            np.array(inputs_, dtype=np.int32)
            for ext, inputs_ in zip(self.encoder_names, inputs)
        ]

        # starts with BOS and ends with EOS, shape is (time, batch_size)
        targets = np.array(targets).T

        return inputs, targets, input_length
