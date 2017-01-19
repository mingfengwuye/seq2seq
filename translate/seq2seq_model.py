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

from translate import utils
from translate import decoders
from collections import namedtuple

from tensorflow.python.ops import variable_scope
from tensorflow.contrib.layers import fully_connected


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

    def __init__(self, encoders, decoder, learning_rate, global_step, max_gradient_norm, num_samples=512,
                 dropout_rate=0.0, freeze_variables=None, lm_weight=None, max_output_len=50, attention=True,
                 feed_previous=0.0, optimizer='sgd', max_input_len=None, decode_only=False, len_normalization=1.0,
                 reinforce_baseline=True, reinforce_reward='sentence_bleu', **kwargs):
        self.lm_weight = lm_weight
        self.encoders = encoders
        self.decoder = decoder

        self.learning_rate = learning_rate
        self.global_step = global_step

        self.encoder_count = len(encoders)
        self.trg_vocab_size = decoder.vocab_size
        self.trg_cell_size = decoder.cell_size
        self.binary_input = [encoder.name for encoder in encoders if encoder.binary]

        self.max_output_len = max_output_len
        self.max_input_len = max_input_len
        self.len_normalization = len_normalization

        # if we use sampled softmax, we need an output projection
        # sampled softmax only makes sense if we sample less than vocabulary size
        if num_samples == 0 or num_samples >= self.trg_vocab_size:
            output_projection = None
            softmax_loss_function = None
        else:
            with tf.device('/cpu:0'):
                with variable_scope.variable_scope('decoder_{}'.format(decoder.name)):
                    w = decoders.get_variable_unsafe('proj_w', [self.trg_cell_size, self.trg_vocab_size])
                    w_t = tf.transpose(w)
                    b = decoders.get_variable_unsafe('proj_b', [self.trg_vocab_size])
                output_projection = (w, b)

            def softmax_loss_function(inputs, labels):
                with tf.device('/cpu:0'):
                    labels = tf.reshape(labels, [-1, 1])
                    return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples, self.trg_vocab_size)

        if dropout_rate > 0:
            self.dropout = tf.Variable(1 - dropout_rate, trainable=False, name='dropout_keep_prob')
            self.dropout_off = self.dropout.assign(1.0)
            self.dropout_on = self.dropout.assign(1 - dropout_rate)
        else:
            self.dropout = None

        self.feed_previous = tf.constant(feed_previous, dtype=tf.float32)
        self.use_reinforce = tf.constant(False, dtype=tf.bool)

        self.reinforce_reward = getattr(utils, reinforce_reward)

        self.encoder_inputs = []
        self.encoder_input_length = []

        self.extensions = [encoder.name for encoder in encoders] + [decoder.name]
        self.encoder_names = [encoder.name for encoder in encoders]
        self.decoder_name = decoder.name
        self.extensions = self.encoder_names + [self.decoder_name]

        for encoder in self.encoders:
            if encoder.binary:
                placeholder = tf.placeholder(tf.float32, shape=[None, None, encoder.embedding_size],
                                             name='encoder_{}'.format(encoder.name))
            else:
                # batch_size x time
                placeholder = tf.placeholder(tf.int32, shape=[None, None],
                                             name='encoder_{}'.format(encoder.name))

            self.encoder_inputs.append(placeholder)
            self.encoder_input_length.append(
                tf.placeholder(tf.int64, shape=[None], name='encoder_{}_length'.format(encoder.name))
            )

        # time x batch_size
        self.decoder_inputs = tf.placeholder(tf.int32, shape=[None, None],
                                             name='decoder_{}'.format(self.decoder.name))
        self.target_weights = tf.placeholder(tf.float32, shape=[None, None],
                                             name='weight_{}'.format(self.decoder.name))
        self.targets = tf.placeholder(tf.int32, shape=[None, None], name='target_{}'.format(self.decoder.name))

        self.decoder_input_length = tf.placeholder(tf.int64, shape=[None],
                                                   name='decoder_{}_length'.format(decoder.name))
        self.reward = tf.placeholder(tf.float32, shape=[None], name='reward')

        parameters = dict(encoders=encoders, decoder=decoder, dropout=self.dropout,
                          output_projection=output_projection,
                          encoder_input_length=self.encoder_input_length)

        self.attention_states, self.encoder_state = decoders.multi_encoder(self.encoder_inputs, **parameters)
        decoder = decoders.attention_decoder if attention else decoders.decoder

        self.outputs, self.attention_weights, self.decoder_states, self.beam_tensors = decoder(
            attention_states=self.attention_states, initial_state=self.encoder_state,
            decoder_inputs=self.decoder_inputs, feed_previous=self.feed_previous,
            decoder_input_length=self.decoder_input_length, **parameters
        )

        self.xent_loss = decoders.sequence_loss(
            logits=self.outputs, targets=self.targets, weights=self.target_weights,
            softmax_loss_function=softmax_loss_function
        )

        def tensor_prod(x, w, b):
            shape = tf.shape(x)
            x = tf.reshape(x, tf.pack([tf.mul(shape[0], shape[1]), shape[2]]))
            x = tf.matmul(x, w) + b
            x = tf.reshape(x, tf.pack([shape[0], shape[1], b.get_shape()[0]]))
            return x

        if output_projection is not None:
            w, b = output_projection
            self.outputs = tensor_prod(self.outputs, w, b)

        time_steps = tf.shape(self.decoder_states)[0]
        batch_size = tf.shape(self.decoder_states)[1]

        output_size = self.outputs.get_shape()[2]
        outputs = tf.reshape(self.outputs, tf.pack([time_steps * batch_size, output_size]))
        sampled_output = tf.multinomial(tf.log(tf.nn.softmax(outputs)), num_samples=1)
        self.sampled_output = tf.reshape(sampled_output, tf.pack([time_steps, batch_size]))

        if reinforce_baseline:
            state_size = self.decoder_states.get_shape()[2]
            states = tf.reshape(self.decoder_states, shape=tf.pack([time_steps * batch_size, state_size]))

            self.baseline = fully_connected(tf.stop_gradient(states), num_outputs=1, activation_fn=None,
                                            scope='reward_baseline',
                                            weights_initializer=tf.constant_initializer(0.0),
                                            biases_initializer=tf.constant_initializer(0.01))
            self.baseline = tf.reshape(self.baseline, shape=tf.pack([time_steps, batch_size]))

            reward = tf.reshape(tf.tile(self.reward, [time_steps]),
                                shape=tf.pack([time_steps, batch_size]))
            reward = reward - tf.stop_gradient(self.baseline)   # reward - baseline, or baseline - reward?
            self.baseline_loss = decoders.baseline_loss(baseline=self.baseline, reward=self.reward,
                                                        weights=self.target_weights)
        else:
            reward = self.reward
            self.baseline_loss = tf.constant(0.0)

        self.reinforce_loss = decoders.sequence_loss(
            logits=self.outputs, targets=self.targets, weights=self.target_weights,
            softmax_loss_function=softmax_loss_function, reward=reward
        )

        if not decode_only:
            # gradients and SGD update operation for training the model
            if freeze_variables is None:
                freeze_variables = []

            # compute gradient only for variables that are not frozen
            frozen_parameters = [var.name for var in tf.trainable_variables()
                                 if any(re.match(var_, var.name) for var_ in freeze_variables)]
            if frozen_parameters:
                utils.debug('frozen parameters: {}'.format(', '.join(frozen_parameters)))
            params = [var for var in tf.trainable_variables() if var.name not in frozen_parameters]

            sgd_opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

            if optimizer.lower() == 'adadelta':
                # same epsilon and rho as Bahdanau et al. 2015
                opt = tf.train.AdadeltaOptimizer(learning_rate=1.0, epsilon=1e-06, rho=0.95)
            elif optimizer.lower() == 'adam':
                opt = tf.train.AdamOptimizer(learning_rate=0.001)
            else:
                opt = sgd_opt

            self.loss = tf.cond(
                self.use_reinforce,
                lambda: self.reinforce_loss,
                lambda: self.xent_loss
            )

            gradients = tf.gradients(self.loss, params)
            clipped_gradients, self.gradient_norms = tf.clip_by_global_norm(gradients, max_gradient_norm)

            self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

            if opt is sgd_opt:
                self.sgd_updates = self.updates
            else:
                self.sgd_updates = sgd_opt.apply_gradients(zip(clipped_gradients, params),
                                                           global_step=self.global_step)

            if reinforce_baseline:
                baseline_opt = sgd_opt
                baseline_gradients = tf.gradients(self.baseline_loss, params)
                clipped_gradients, _= tf.clip_by_global_norm(baseline_gradients, max_gradient_norm)
                self.baseline_updates = baseline_opt.apply_gradients(zip(clipped_gradients, params))
            else:
                self.baseline_updates = tf.constant(0.0)   # dummy tensor

        self.beam_output = tf.nn.softmax(self.outputs[0,:,:])

    def step(self, session, data, update_model=True, align=False, **kwargs):
        if self.dropout is not None:
            session.run(self.dropout_on)

        batch = self.get_batch(data)
        encoder_inputs, decoder_inputs, targets, target_weights, encoder_input_length, decoder_input_length = batch

        batch_size = targets.shape[1]

        input_feed = {
            self.target_weights: target_weights,
            self.decoder_inputs: decoder_inputs,
            self.decoder_input_length: decoder_input_length,
            self.targets: targets,
            self.reward: np.zeros((batch_size,)),
        }

        for i in range(self.encoder_count):
            input_feed[self.encoder_input_length[i]] = encoder_input_length[i]
            input_feed[self.encoder_inputs[i]] = encoder_inputs[i]

        output_feed = {'loss': self.loss, 'gradient': self.gradient_norms}
        if update_model:
            output_feed['updates'] = self.updates
        if align:
            output_feed['attn_weights'] = self.attention_weights

        res = session.run(output_feed, input_feed)
        return namedtuple('output', 'loss attn_weights')(res['loss'], res.get('attn_weights'))

    def reinforce_step(self, session, data, update_model=True, update_baseline=True, **kwargs):
        if self.dropout is not None:
            session.run(self.dropout_off)

        batch = self.get_batch(data)
        encoder_inputs, decoder_inputs, targets, target_weights, encoder_input_length, decoder_input_length = batch

        input_feed = {
            self.decoder_inputs: decoder_inputs,
            self.decoder_input_length: decoder_input_length,
        }

        for i in range(self.encoder_count):
            input_feed[self.encoder_input_length[i]] = encoder_input_length[i]
            input_feed[self.encoder_inputs[i]] = encoder_inputs[i]

        output_feed = [self.sampled_output, self.decoder_states]
        outputs, decoder_states = session.run(output_feed, input_feed)

        outputs_ = outputs.T
        targets_ = targets.T
        lengths_ = decoder_input_length

        scores = []
        weights = []
        for output_, target_, length_ in zip(outputs_, targets_, lengths_):
            target_ = target_[:length_]    # includes first EOS (good idea?)

            i, = np.where(output_ == utils.EOS_ID)  # array of indices whose value is EOS_ID

            time_steps = output_.shape[0]
            if len(i) > 0:
                i = i[0]  # index of the first EOS symbol
                weights_ = (np.arange(time_steps) <= i).astype(np.float32)
                output_ = output_[:i + 1]  # includes first EOS
            else:
                weights_ = np.ones(time_steps)

            weights.append(weights_)

            score = self.reinforce_reward(output_, target_)
            scores.append(score)

        weights = np.transpose(weights)

        if self.dropout is not None:
            session.run(self.dropout_on)

        input_feed = {
            **input_feed,
            self.reward: scores,
            self.target_weights: weights,
            self.targets: outputs,
            self.use_reinforce: True,
        }

        output_feed = {'loss': self.xent_loss, 'baseline_loss': self.baseline_loss}

        if update_model:
            output_feed['updates'] = self.updates

        if update_baseline:
            output_feed['baseline_updates'] = self.baseline_updates

        res = session.run(output_feed, input_feed)

        return namedtuple('output', 'loss baseline_loss')(res['loss'], res['baseline_loss'])

    def greedy_decoding(self, session, token_ids):
        if self.dropout is not None:
            session.run(self.dropout_off)

        batch = self.get_batch([token_ids + [[]]], decoding=True)
        encoder_inputs, decoder_inputs, targets, target_weights, encoder_input_length, decoder_input_length = batch

        input_feed = {
            self.target_weights: target_weights,
            self.decoder_inputs: decoder_inputs,
            self.decoder_input_length: decoder_input_length,
            self.targets: targets,
            self.feed_previous: 1.0
        }

        for i in range(self.encoder_count):
            input_feed[self.encoder_input_length[i]] = encoder_input_length[i]
            input_feed[self.encoder_inputs[i]] = encoder_inputs[i]

        outputs, attn_weights = session.run([self.outputs, self.attention_weights], input_feed)
        return [int(np.argmax(logit, axis=1)) for logit in outputs], attn_weights  # greedy decoder

    def beam_search_decoding(self, session, token_ids, beam_size, ngrams=None):
        if not isinstance(session, list):
            session = [session]

        if self.dropout is not None:
            for session_ in session:
                session_.run(self.dropout_off)

        data = [token_ids + [[]]]
        batch = self.get_batch(data, decoding=True)
        encoder_inputs, decoder_inputs, targets, target_weights, encoder_input_length, _ = batch
        input_feed = {}

        for i in range(self.encoder_count):
            input_feed[self.encoder_input_length[i]] = encoder_input_length[i]
            input_feed[self.encoder_inputs[i]] = encoder_inputs[i]

        output_feed = [self.encoder_state] + self.attention_states
        res = [session_.run(output_feed, input_feed) for session_ in session]
        state, attn_states = list(zip(*[(res_[0], res_[1:]) for res_ in res]))

        decoder_input = decoder_inputs[0]  # BOS symbol

        finished_hypotheses = []
        finished_scores = []

        hypotheses = [[]]
        scores = np.zeros([1], dtype=np.float32)

        # for initial state projection
        state = [session_.run(self.beam_tensors.state, {self.encoder_state: state_})
                 for session_, state_ in zip(session, state)]
        output = None

        for i in range(self.max_output_len):
            # each session/model has its own input and output
            # in beam-search decoder, we only feed the first input
            batch_size = decoder_input.shape[0]

            input_feed = [
                {self.beam_tensors.state: state_,
                 self.decoder_inputs: np.reshape(decoder_input, [1, batch_size]),
                 self.decoder_input_length: [1] * batch_size,
                }
                for state_ in state
            ]
            
            for feed in input_feed:
                for i in range(self.encoder_count):
                    feed[self.encoder_input_length[i]] = encoder_input_length[i]

            if i > 0:
                for input_feed_, output_ in zip(input_feed, output):
                    input_feed_[self.beam_tensors.output] = output_

            for input_feed_, attn_states_ in zip(input_feed, attn_states):
                for i in range(self.encoder_count):
                    input_feed_[self.attention_states[i]] = attn_states_[i].repeat(batch_size, axis=0)

            output_feed = namedtuple('beam_output', 'output decoder_output decoder_state')(
                self.beam_tensors.new_output,
                self.beam_output,
                self.beam_tensors.new_state,
            )

            res = [session_.run(output_feed, input_feed_) for session_, input_feed_ in zip(session, input_feed)]

            res_transpose = list(
                zip(*[(res_.output, res_.decoder_output, res_.decoder_state) for res_ in res])
            )

            output, decoder_output, decoder_state = res_transpose
            # hypotheses, list of tokens ids of shape (beam_size, previous_len)
            # decoder_output, shape=(beam_size, trg_vocab_size)
            # decoder_state, shape=(beam_size, cell.state_size)
            # attention_weights, shape=(beam_size, max_len)

            if ngrams is not None:
                lm_score = []
                lm_order = len(ngrams)

                for hypothesis in hypotheses:
                    # not sure about this (should we put <s> at the beginning?)
                    hypothesis = [utils.BOS_ID] + hypothesis
                    history = hypothesis[1 - lm_order:]
                    score_ = []

                    for token_id in range(self.trg_vocab_size):
                        # if token is not in unigrams, this means that either there is something
                        # wrong with the ngrams (e.g. trained on wrong file),
                        # or trg_vocab_size is larger than actual vocabulary
                        if (token_id,) not in ngrams[0]:
                            prob = float('-inf')
                        elif token_id == utils.BOS_ID:
                            prob = float('-inf')
                        else:
                            prob = utils.estimate_lm_score(history + [token_id], ngrams)
                        score_.append(prob)

                    lm_score.append(score_)
                lm_score = np.array(lm_score, dtype=np.float32)
                lm_weight = self.lm_weight or 0.2
                weights = [(1 - lm_weight) / len(session)] * len(session) + [lm_weight]
            else:
                lm_score = np.zeros((1, self.trg_vocab_size))
                weights = None

            # FIXME: divide by zero encountered in log
            scores_ = scores[:, None] - np.average([np.log(decoder_output_) for decoder_output_ in decoder_output] +
                                                   [lm_score], axis=0, weights=weights)
            scores_ = scores_.flatten()
            flat_ids = np.argsort(scores_)[:beam_size]

            token_ids_ = flat_ids % self.trg_vocab_size
            hyp_ids = flat_ids // self.trg_vocab_size

            new_hypotheses = []
            new_scores = []
            new_state = [[] for _ in session]
            new_output = [[] for _ in session]
            new_input = []

            for flat_id, hyp_id, token_id in zip(flat_ids, hyp_ids, token_ids_):
                hypothesis = hypotheses[hyp_id] + [token_id]
                score = scores_[flat_id]

                if token_id == utils.EOS_ID:
                    # early stop: hypothesis is finished, it is thus unnecessary to keep expanding it
                    beam_size -= 1  # number of possible hypotheses is reduced by one
                    finished_hypotheses.append(hypothesis)
                    finished_scores.append(score)
                else:
                    new_hypotheses.append(hypothesis)

                    for session_id, decoder_state_, in enumerate(decoder_state):
                        new_state[session_id].append(decoder_state_[hyp_id])
                        new_output[session_id].append(output[session_id][hyp_id])

                    new_scores.append(score)
                    new_input.append(token_id)

            hypotheses = new_hypotheses
            state = [np.array(new_state_) for new_state_ in new_state]
            output = [np.array(new_output_) for new_output_ in new_output]
            scores = np.array(new_scores)
            decoder_input = np.array(new_input, dtype=np.int32)

            if beam_size <= 0:
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
        encoder_inputs = [[] for _ in range(self.encoder_count)]
        encoder_input_length = [[] for _ in range(self.encoder_count)]
        decoder_inputs = []
        decoder_input_length = []

        # maximum input length of each encoder in this batch
        max_input_len = [max(len(data_[i]) for data_ in data) for i in range(self.encoder_count)]
        if self.max_input_len is not None:
            max_input_len = [min(len_, self.max_input_len) for len_ in max_input_len]
        # maximum output length in this batch
        max_output_len = min(max(len(data_[-1]) for data_ in data), self.max_output_len)

        for *src_sentences, trg_sentence in data:
            for i, (encoder, src_sentence) in enumerate(zip(self.encoders, src_sentences)):
                if encoder.binary:
                    # when using binary input, the input sequence is a sequence of vectors,
                    # instead of a sequence of indices
                    pad = np.zeros([encoder.embedding_size], dtype=np.float32)
                else:
                    pad = utils.EOS_ID

                # pad sequences so that all sequences in the same batch have the same length
                src_sentence = src_sentence[:max_input_len[i]]
                encoder_pad = [pad] * (1 + max_input_len[i] - len(src_sentence))

                encoder_inputs[i].append(src_sentence + encoder_pad)
                encoder_input_length[i].append(len(src_sentence) + 1)

            trg_sentence = trg_sentence[:max_output_len]
            if decoding:
                decoder_input_length.append(self.max_output_len)
                decoder_inputs.append([utils.BOS_ID] + [utils.EOS_ID] * self.max_output_len)
            else:
                decoder_pad_size = max_output_len - len(trg_sentence)
                decoder_input_length.append(len(trg_sentence) + 1)
                trg_sentence = [utils.BOS_ID] + trg_sentence + [utils.EOS_ID] + [-1] * decoder_pad_size
                decoder_inputs.append(trg_sentence)

        # convert lists to numpy arrays
        encoder_input_length = [np.array(input_length_, dtype=np.int32) for input_length_ in encoder_input_length]
        decoder_input_length = np.array(decoder_input_length, dtype=np.int32)
        batch_encoder_inputs = [
            np.array(encoder_inputs_, dtype=(np.float32 if ext in self.binary_input else np.int32))
            for ext, encoder_inputs_ in zip(self.encoder_names, encoder_inputs)
        ]  # for binary input, the data type is float32

        # time-major vectors: shape is (time, batch_size)
        batch_decoder_inputs = np.array(decoder_inputs)[:, :-1].T  # with BOS symbol, without EOS symbol
        batch_targets = np.array(decoder_inputs)[:, 1:].T  # without BOS symbol, with EOS symbol
        batch_weights = (batch_targets != -1).astype(np.float32)  # PAD symbols don't count for training

        batch_decoder_inputs[batch_decoder_inputs == -1] = utils.EOS_ID
        batch_targets[batch_targets == -1] = utils.EOS_ID

        return (batch_encoder_inputs,
                batch_decoder_inputs,
                batch_targets,
                batch_weights,
                encoder_input_length,
                decoder_input_length)
