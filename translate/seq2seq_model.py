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
                 freeze_variables=None, lm_weight=None, max_output_len=50, feed_previous=0.0,
                 optimizer='sgd', max_input_len=None, decode_only=False, len_normalization=1.0,
                 reinforce_baseline=True, softmax_temperature=1.0, loss_function='xent', rollouts=None,
                 partial_rewards=False, use_edits=False, sub_op=False, **kwargs):
        self.lm_weight = lm_weight
        self.encoders = encoders
        self.decoder = decoder
        self.oracle = decoder.oracle

        self.learning_rate = learning_rate
        self.global_step = global_step

        self.encoder_count = len(encoders)
        self.trg_vocab_size = decoder.vocab_size
        self.trg_cell_size = decoder.cell_size
        self.binary_input = [encoder.name for encoder in encoders if encoder.binary]

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


        # starts with BOS, and ends with EOS  (time x batch_size)
        self.targets = tf.placeholder(tf.int32, shape=[None, None], name='target_{}'.format(self.decoder.name))

        if self.oracle:
            self.decoder_inputs = tf.placeholder(tf.int32, shape=[None, None], name='decoder_inputs_{}'.format(
                self.decoder.name))
        else:
            self.decoder_inputs = self.targets[:-1,:]

        self.target_weights = decoders.get_weights(self.targets[1:,:], utils.EOS_ID, time_major=True,
                                                   include_first_eos=True)
        self.target_length = tf.reduce_sum(self.target_weights, axis=0)

        if loss_function == 'xent' or decode_only:  # FIXME: use tensor instead
            self.rollouts = None
        else:
            self.rollouts = rollouts

        self.partial_rewards = partial_rewards

        parameters = dict(encoders=encoders, decoder=decoder, dropout=self.dropout,
                          encoder_input_length=self.encoder_input_length, rollouts=1,
                          use_edits=use_edits, sub_op=sub_op)

        self.attention_states, self.encoder_state = decoders.multi_encoder(self.encoder_inputs, **parameters)

        # (self.outputs, self.attention_weights, self.decoder_outputs, self.beam_tensors,
        #  self.sampled_output, self.states) = decoders.attention_decoder(
        #     attention_states=self.attention_states, initial_state=self.encoder_state,
        #     feed_previous=self.feed_previous, decoder_inputs=self.decoder_inputs,
        #     decoder_input_length=self.target_length, feed_argmax=self.feed_argmax, **parameters
        # )

        self.outputs = decoders.attention_decoder(
            attention_states=self.attention_states, initial_state=self.encoder_state,
            feed_previous=self.feed_previous, decoder_inputs=self.decoder_inputs,
            decoder_input_length=self.target_length, feed_argmax=self.feed_argmax, **parameters
        )

        self.beam_tensors = decoders.beam_attention_decoder(
            attention_states=self.attention_states, initial_state=self.encoder_state,
            feed_previous=self.feed_previous, decoder_inputs=self.decoder_inputs,
            decoder_input_length=self.target_length, feed_argmax=self.feed_argmax, **parameters
        )

        self.beam_output = decoders.softmax(self.outputs[0, :, :], temperature=softmax_temperature)

        optimizers = self.get_optimizers(optimizer, learning_rate)

        self.xent_loss, self.reinforce_loss, self.baseline_loss = None, None, None
        self.update_op, self.sgd_update_op, self.baseline_update_op = None, None, None
        self.rewards = None

        if loss_function == 'xent':
            self.init_xent(optimizers, decode_only)
        else:
            self.init_reinforce(optimizers, reinforce_baseline, decode_only)
            self.init_xent(optimizers, decode_only=True)   # used for eval

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

    def init_reinforce(self, optimizers, reinforce_baseline=True, decode_only=False):
        self.rewards = tf.placeholder(tf.float32, [None, None], 'rewards')

        if reinforce_baseline:
            reward = decoders.reinforce_baseline(self.decoder_outputs, self.rewards)
            weights = decoders.get_weights(self.sampled_output, utils.EOS_ID, time_major=True,
                                           include_first_eos=False)
            self.baseline_loss = decoders.baseline_loss(reward=reward, weights=weights)
        else:
            reward = self.rewards
            self.baseline_loss = tf.constant(0.0)

        weights = decoders.get_weights(self.sampled_output, utils.EOS_ID, time_major=True,
                                       include_first_eos=True)   # FIXME: True or False?
        self.reinforce_loss = decoders.sequence_loss(logits=self.outputs, targets=self.sampled_output,
                                                     weights=weights, reward=reward)

        if not decode_only:
            self.update_op, self.sgd_update_op = self.get_update_op(self.reinforce_loss,
                                                                    optimizers,
                                                                    self.global_step)

            if reinforce_baseline:
                baseline_opt = tf.train.AdamOptimizer(learning_rate=0.001)
                self.baseline_update_op, = self.get_update_op(self.baseline_loss, [baseline_opt])
            else:
                self.baseline_update_op = tf.constant(0.0)   # dummy tensor

    def step(self, session, data, update_model=True, align=False, use_sgd=False, **kwargs):
        if self.dropout is not None:
            session.run(self.dropout_on)

        batch = self.get_batch(data)
        encoder_inputs, targets, encoder_input_length = batch

        input_feed = {}

        if self.oracle:
            decoder_inputs = targets[:-1,:]    # feed decoder with full op
            input_feed[self.decoder_inputs] = decoder_inputs

            # learn to predict op type
            targets[targets >= len(utils._START_VOCAB)] = utils.INS_ID   # no SUB op for now
            targets[targets == utils.UNK_ID] = utils.INS_ID

        input_feed[self.targets] = targets

        # if not update_model:
        #     import ipdb; ipdb.set_trace()

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


    def reinforce_step(self, session, data, update_model=True, update_baseline=True,
                       use_sgd=False, reward_function=None, use_edits=False, vocabs=None, **kwargs):
        assert vocabs or not use_edits

        if vocabs:
            src_vocab = vocabs[0]
            trg_vocab = vocabs[-1]

        if self.dropout is not None:
            session.run(self.dropout_off)

        batch = self.get_batch(data)
        encoder_inputs, targets, encoder_input_length = batch

        time_steps = targets.shape[0]
        batch_size = targets.shape[1]

        max_output_len = min(self.max_output_len, int(1.5 * time_steps))

        if time_steps <= max_output_len:
            targets = np.pad(targets,
                [(0, max_output_len + 1 - time_steps), (0, 0)],
                mode='constant', constant_values=utils.EOS_ID,)
        target_length = [max_output_len] * batch_size

        input_feed = {
            self.targets: targets,
            self.target_length: target_length,
            self.feed_previous: 1.0,
            self.feed_argmax: False   # sample from softmax
        }

        for i in range(self.encoder_count):
            input_feed[self.encoder_input_length[i]] = encoder_input_length[i]
            input_feed[self.encoder_inputs[i]] = encoder_inputs[i]

        sampled_output, outputs, states = session.run([self.sampled_output, self.outputs, self.states],
                                                       input_feed)

        time_steps = sampled_output.shape[0]

        if reward_function is None:
            reward_function = 'sentence_bleu'

        reward_function = getattr(evaluation, reward_function)

        def compute_reward(output, target, source, partial=False):
            j, = np.where(output == utils.EOS_ID)  # array of indices whose value is EOS_ID
            if len(j) > 0:
                output = output[:j[0]]

            j, = np.where(target == utils.EOS_ID)
            if len(j) > 0:
                target = target[:j[0]]

            j, = np.where(source == utils.EOS_ID)
            if len(j) > 0:
                source = source[:j[0]]

            if use_edits:   # use source sentence and sequence of edits to reconstruct output and target sequence
                # It makes more sense to compute the reward on the reconstructed sequences, than on the sequences
                # of edits.
                output = utils.reverse_edit_ids(source, output, src_vocab, trg_vocab)
                target = utils.reverse_edit_ids(source, target, src_vocab, trg_vocab)

            if partial:
                reward = [reward_function(output[:i + 1], target) for i in range(len(output))]
                reward = [0] + reward
                reward += [reward[-1]] * (time_steps - len(reward) + 1)
                reward = np.array(reward)
                return reward[1:] - reward[:-1]
            else:
                return reward_function(output, target)

        def compute_rewards(outputs, targets, partial=False):
            return np.array([compute_reward(output, target, source, partial=partial)
                             for output, target, source in zip(outputs.T, targets.T, encoder_inputs[0])])

        targets = targets[1:]

        if self.rollouts is not None and self.rollouts > 1:
            rewards = []

            for i in range(time_steps):

                if i == time_steps - 1:
                    rewards.append(compute_rewards(sampled_output, targets))
                    continue

                reward = 0

                for _ in range(self.rollouts):
                    prefix = sampled_output[:i + 1]

                    input_ = np.expand_dims(sampled_output[i], axis=0)
                    targets_ = targets[i + 1:]
                    targets_ = np.concatenate([input_, targets_], axis=0)
                    target_length_ = [time_steps - i - 1] * batch_size

                    input_feed_ = dict(input_feed)
                    input_feed_[self.targets] = targets_
                    input_feed_[self.target_length] = target_length_
                    input_feed_[self.beam_tensors.state] = states[i]

                    outputs_ = session.run(self.sampled_output, input_feed_)
                    outputs_ = np.concatenate([prefix, outputs_], axis=0)

                    reward += compute_rewards(outputs_, targets) / self.rollouts

                rewards.append(reward)

            rewards = np.array(rewards)
        elif self.partial_rewards:
            rewards = compute_rewards(sampled_output, targets, partial=True).T
        else:
            rewards = compute_rewards(sampled_output, targets)
            rewards = np.stack([rewards] * time_steps)

        input_feed[self.rewards] = rewards
        input_feed[self.outputs] = outputs
        input_feed[self.sampled_output] = sampled_output

        output_feed = {'loss': self.reinforce_loss, 'baseline_loss': self.baseline_loss}

        if update_model:
            output_feed['updates'] = self.sgd_update_op if use_sgd else self.update_op

        if update_baseline:
            output_feed['baseline_updates'] = self.baseline_update_op

        res = session.run(output_feed, input_feed)

        return namedtuple('output', 'loss baseline_loss')(res['loss'], res['baseline_loss'])

    def greedy_decoding(self, session, token_ids):
        if self.dropout is not None:
            session.run(self.dropout_off)

        token_ids = [token_ids_ + [[]] for token_ids_ in token_ids]

        batch = self.get_batch(token_ids, decoding=True)
        encoder_inputs, targets, encoder_input_length = batch
        # utils.debug(encoder_inputs[0][5])

        input_feed = {self.targets: targets, self.feed_previous: 1.0}

        for i in range(self.encoder_count):
            input_feed[self.encoder_input_length[i]] = encoder_input_length[i]
            input_feed[self.encoder_inputs[i]] = encoder_inputs[i]

        outputs = session.run(self.outputs, input_feed)

        return np.argmax(outputs, axis=2).T

    def beam_search_decoding_old(self, session, token_ids, beam_size, ngrams=None, early_stopping=True,
                             use_edits=False):
        if not isinstance(session, list):
            session = [session]

        if self.dropout is not None:
            for session_ in session:
                session_.run(self.dropout_off)

        data = [token_ids + [[]]]
        batch = self.get_batch(data, decoding=True)
        encoder_inputs, targets, encoder_input_length = batch
        input_feed = {}

        for i in range(self.encoder_count):
            input_feed[self.encoder_input_length[i]] = encoder_input_length[i]
            input_feed[self.encoder_inputs[i]] = encoder_inputs[i]

        output_feed = [self.encoder_state] + self.attention_states
        res = [session_.run(output_feed, input_feed) for session_ in session]
        state, attn_states = list(zip(*[(res_[0], res_[1:]) for res_ in res]))

        targets = targets[0]  # BOS symbol

        edit_pos = [np.zeros([1], dtype=np.int32) for _ in session]

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
            batch_size = targets.shape[0]
            targets = np.reshape(targets, [1, batch_size])
            targets = np.concatenate([targets, np.ones(targets.shape) * utils.EOS_ID])

            input_feed = [
                {self.beam_tensors.state: state_,
                 self.targets: targets,
                 self.target_length: [1] * batch_size,
                 self.beam_tensors.edit_pos: edit_pos_
                }
                for state_, edit_pos_ in zip(state, edit_pos)
            ]
            
            for feed in input_feed:
                for j in range(self.encoder_count):
                    feed[self.encoder_input_length[j]] = encoder_input_length[j]

            if i > 0:
                for input_feed_, output_ in zip(input_feed, output):
                    input_feed_[self.beam_tensors.output] = output_

            for input_feed_, attn_states_ in zip(input_feed, attn_states):
                for j in range(self.encoder_count):
                    input_feed_[self.attention_states[j]] = attn_states_[j].repeat(batch_size, axis=0)

            output_feed = namedtuple('beam_output', 'output state proba edit_pos')(
                self.beam_tensors.new_output,
                self.beam_tensors.new_state,
                self.beam_output,
                self.beam_tensors.new_edit_pos
            )

            res = [session_.run(output_feed, input_feed_) for session_, input_feed_ in zip(session, input_feed)]

            res_transpose = list(
                zip(*[(res_.output, res_.state, res_.proba, res_.edit_pos) for res_ in res])
            )

            output, state, proba, edit_pos = res_transpose
            # hypotheses, list of tokens ids of shape (beam_size, previous_len)
            # proba, shape=(beam_size, trg_vocab_size)
            # state, shape=(beam_size, cell.state_size)
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

            proba = [np.maximum(proba_, 1e-10) for proba_ in proba]
            scores_ = scores[:, None] - np.average([np.log(proba_) for proba_ in proba] +
                                                   [lm_score], axis=0, weights=weights)
            scores_ = scores_.flatten()
            flat_ids = np.argsort(scores_)

            token_ids_ = flat_ids % self.trg_vocab_size
            hyp_ids = flat_ids // self.trg_vocab_size

            new_hypotheses = []
            new_scores = []
            new_state = [[] for _ in session]
            new_edit_pos = [[] for _ in session]
            new_output = [[] for _ in session]
            new_input = []
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

                    for session_id, state_, in enumerate(state):
                        new_state[session_id].append(state_[hyp_id])
                        new_output[session_id].append(output[session_id][hyp_id])
                        new_edit_pos[session_id].append(edit_pos[session_id][hyp_id])

                    new_scores.append(score)
                    new_input.append(token_id)

                if len(new_hypotheses) == beam_size:
                    break

            beam_size = new_beam_size
            hypotheses = new_hypotheses
            state = [np.array(new_state_) for new_state_ in new_state]
            edit_pos = [np.array(new_edit_pos_) for new_edit_pos_ in new_edit_pos]
            output = [np.array(new_output_) for new_output_ in new_output]
            scores = np.array(new_scores)
            targets = np.array(new_input, dtype=np.int32)

            if beam_size <= 0:
                break

        # hypotheses += finished_hypotheses
        # scores = np.concatenate([scores, finished_scores])

        hypotheses = [
            list(itertools.takewhile(lambda x: x != utils.EOS_ID, hyp))
            for hyp in hypotheses
        ]  # FIXME: counting EOS score

        if use_edits:  # TODO: reverse edits
            hypothesis_len = [len(hyp) - hyp.count(utils.DEL_ID) for hyp in hypotheses]
            # TODO: penalty if hypothesis is broken
            # TODO: PEP (post-editing penalty) = -1 for each new word w.r.t the input

            n = len(encoder_inputs[0][0])

            penalty = np.array([
                abs(n - hyp.count(utils.DEL_ID) - hyp.count(utils.KEEP_ID) - hyp.count(utils.SUB_ID))
                for hyp in hypotheses
            ])

            # scores += penalty
        else:
            hypothesis_len = map(len, hypotheses)

        if self.len_normalization > 0:  # normalize score by length (to encourage longer sentences)
            scores /= [len_ ** self.len_normalization for len_ in hypothesis_len]

        # sort best-list by score
        sorted_idx = np.argsort(scores)
        hypotheses = np.array(hypotheses)[sorted_idx].tolist()
        scores = scores[sorted_idx].tolist()
        return hypotheses, scores

    def greedy_step_by_step_decoding(self, session, token_ids):
        if self.dropout is not None:
            session.run(self.dropout_off)

        data = [
            ids + [[]] if len(ids) == self.encoder_count else ids
            for ids in token_ids
        ]

        if self.oracle:
            encoder_inputs, references, encoder_input_length =  self.get_batch(data, decoding=False)
            insertions = [[i for i in reference if i >= len(utils._START_VOCAB)]
                          for reference in references.T]
        else:
            encoder_inputs, _, encoder_input_length = self.get_batch(data, decoding=True)
            insertions = None

        input_feed = {}
        for i in range(self.encoder_count):
            input_feed[self.encoder_input_length[i]] = encoder_input_length[i]
            input_feed[self.encoder_inputs[i]] = encoder_inputs[i]

        output_feed = [self.encoder_state, self.attention_states, self.beam_tensors.initial_data,
                       self.beam_tensors.initial_output]

        initial_state, attn_states, data, output = session.run(output_feed, input_feed)

        hypotheses = []

        for i in range(self.max_output_len):
            targets = np.argmax(output, axis=1)

            if self.oracle:
                # replace insertions by corresponding word
                new_targets = []
                for j, (target, insertions_) in enumerate(zip(targets, insertions)):
                    if target == utils.INS_ID:
                        try:
                            new_targets.append(insertions_.pop(0))
                        except IndexError:
                            new_targets.append(utils.NONE_ID)
                    else:
                        new_targets.append(target)

                targets = np.array(new_targets)

            new_targets = []
            for i, target in enumerate(targets):
                if target == utils.NONE_ID:   # replace NONE_ID by latest output symbol
                    prev = [hyp[i] for hyp in hypotheses if hyp[i] != utils.NONE_ID]
                    target = prev[-1] if len(prev) > 0 else utils.BOS_ID

                new_targets.append(target)

            hypotheses.append(targets)

            # early stopping if all hypotheses are finished
            if all(utils.EOS_ID in hyp for hyp in hypotheses):
                break

            targets = new_targets

            input_feed = {
                self.beam_tensors.data: data,
                self.beam_tensors.decoder_input: targets,
                self.attention_states: attn_states,
                self.encoder_state: initial_state
            }

            for j in range(self.encoder_count):
                input_feed[self.encoder_input_length[j]] = encoder_input_length[j]

            output_feed = [self.beam_tensors.new_data,
                           self.beam_tensors.output]

            data, output = session.run(output_feed, input_feed)

        hypotheses = np.array(hypotheses).T
        hypotheses = [[i for i in hypothesis if i != utils.BOS_ID and i != utils.NONE_ID]
                      for hypothesis in hypotheses]

        return hypotheses

    def beam_search_decoding(self, session, token_ids, beam_size, use_edits=False, early_stopping=True,
                             *args, **kwargs):
        if self.dropout is not None:
            session.run(self.dropout_off)

        assert not self.oracle  # FIXME
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
                self.attention_states: attn_states.repeat(batch_size, axis=1),
                self.encoder_state: initial_state.repeat(batch_size, axis=0)
            }

            for j in range(self.encoder_count):
                input_feed[self.encoder_input_length[j]] = encoder_input_length[j]

            output_feed = [self.beam_tensors.new_data,
                           self.beam_tensors.output]

            data, output = session.run(output_feed, input_feed)

        hypotheses += finished_hypotheses
        scores = np.concatenate([scores, finished_scores])

        if use_edits:
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
                if encoder.binary:
                    # when using binary input, the input sequence is a sequence of vectors,
                    # instead of a sequence of indices
                    pad = np.zeros([encoder.embedding_size], dtype=np.float32)
                else:
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
            np.array(inputs_, dtype=(np.float32 if ext in self.binary_input else np.int32))
            for ext, inputs_ in zip(self.encoder_names, inputs)
        ]  # for binary input, the data type is float32

        # starts with BOS and ends with EOS, shape is (time, batch_size)
        targets = np.array(targets).T

        return inputs, targets, input_length
