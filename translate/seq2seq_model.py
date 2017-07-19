import numpy as np
import tensorflow as tf
import re
import functools

from translate import utils
from translate import models
from translate import evaluation
from collections import namedtuple


class Seq2SeqModel(object):
    def __init__(self, encoders, decoders, learning_rate, global_step, max_gradient_norm, dropout_rate=0.0,
                 freeze_variables=None, feed_previous=0.0, optimizer='sgd', decode_only=False, len_normalization=1.0,
                 name=None, chained_encoders=False, dual_output=False, pred_edits=False, baseline_step=None,
                 use_baseline=True, **kwargs):
        self.encoders = encoders
        self.decoders = decoders
        self.name = name

        self.learning_rate = learning_rate
        self.global_step = global_step
        self.baseline_step = baseline_step
        self.use_baseline = use_baseline

        self.max_output_len = [decoder.max_len for decoder in decoders]
        self.max_input_len = [encoder.max_len for encoder in encoders]
        self.len_normalization = len_normalization

        if dropout_rate > 0:
            self.dropout = tf.Variable(1 - dropout_rate, trainable=False, name='dropout_keep_prob')
            self.dropout_off = self.dropout.assign(1.0)
            self.dropout_on = self.dropout.assign(1 - dropout_rate)
        else:
            self.dropout = None

        self.feed_previous = tf.constant(feed_previous, dtype=tf.float32)
        self.feed_argmax = tf.constant(True, dtype=tf.bool)  # feed with argmax or sample from softmax

        self.encoder_inputs = []
        self.input_weights = []

        self.encoder_inputs = []
        self.encoder_input_length = []
        for encoder in encoders:
            shape = [None, None, encoder.embedding_size] if encoder.binary else [None, None]
            dtype = tf.float32 if encoder.binary else tf.int32
            encoder_input = tf.placeholder(dtype=dtype, shape=shape, name='encoder_{}'.format(encoder.name))
            encoder_input_length = tf.placeholder(dtype=tf.int32, shape=[None],
                                                  name='encoder_input_length_{}'.format(encoder.name))
            self.encoder_inputs.append(encoder_input)
            self.encoder_input_length.append(encoder_input_length)

        # starts with BOS, and ends with EOS
        self.targets = tuple([
            tf.placeholder(tf.int32, shape=[None, None], name='target_{}'.format(decoder.name))
            for decoder in decoders
        ])
        self.rewards = tf.placeholder(tf.float32, shape=[None, None], name='rewards')

        if chained_encoders and pred_edits:
             architecture = models.chained_encoder_decoder    # no REINFORCE for now
        else:
             architecture = models.encoder_decoder
        # elif dual_output or pred_edits:
        #     architecture = models.multi_encoder_decoder

        tensors = architecture(encoders, decoders, self.dropout, self.encoder_inputs, self.targets, self.feed_previous,
                               encoder_input_length=self.encoder_input_length, feed_argmax=self.feed_argmax,
                               rewards=self.rewards, use_baseline=use_baseline, **kwargs)

        (self.losses, self.outputs, self.encoder_state, self.attention_states, self.attention_weights,
         self.beam_tensors, self.samples) = tensors

        self.xent_loss, self.reinforce_loss, self.baseline_loss = self.losses
        self.loss = self.xent_loss   # main loss

        #self.beam_outputs = [models.softmax(outputs_[:, 0, :]) for outputs_ in self.outputs]
        self.beam_output = models.softmax(self.outputs[0][:, 0, :])

        optimizers = self.get_optimizers(optimizer, learning_rate)

        if not decode_only:
            get_update_ops = functools.partial(self.get_update_op, opts=optimizers,
                                               max_gradient_norm=max_gradient_norm, freeze_variables=freeze_variables)

            self.update_ops = utils.AttrDict({
                'xent': get_update_ops(self.xent_loss, global_step=self.global_step),
                'reinforce': get_update_ops(self.reinforce_loss, global_step=self.global_step),
            })

            if use_baseline:
                self.update_ops['baseline'] = get_update_ops(self.baseline_loss, global_step=self.baseline_step)

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

    def get_update_op(self, loss, opts, global_step=None, max_gradient_norm=None, freeze_variables=None):
        freeze_variables = freeze_variables or []

        # compute gradient only for variables that are not frozen
        frozen_parameters = [var.name for var in tf.trainable_variables()
                             if any(re.match(var_, var.name) for var_ in freeze_variables)]
        params = [var for var in tf.trainable_variables() if var.name not in frozen_parameters]
        self.params = params

        gradients = tf.gradients(loss, params)
        if max_gradient_norm:
            gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

        update_ops = []
        for opt in opts:
            with tf.variable_scope('gradients' if self.name is None else 'gradients_{}'.format(self.name)):
                update_op = opt.apply_gradients(list(zip(gradients, params)), global_step=global_step)

            update_ops.append(update_op)

        return update_ops

    def reinforce_step(self, session, data, update_model=True, align=False, use_sgd=False, update_baseline=True,
                       reward_function=None, **kwargs):
        if self.dropout is not None:
            session.run(self.dropout_on)

        encoder_inputs, targets, input_length = self.get_batch(data)
        input_feed = {self.targets: targets, self.feed_argmax: False, self.feed_previous: 1.0}

        for i in range(len(self.encoders)):
            input_feed[self.encoder_inputs[i]] = encoder_inputs[i]
            input_feed[self.encoder_input_length[i]] = input_length[i]

        samples, outputs = session.run([self.samples, self.outputs], input_feed)

        if reward_function is None:
            reward_function = 'sentence_bleu'
        reward_function = getattr(evaluation, reward_function)

        def compute_reward(output, target):
            j, = np.where(output == utils.EOS_ID)  # array of indices whose value is EOS_ID
            if len(j) > 0:
                output = output[:j[0]]

            j, = np.where(target == utils.EOS_ID)
            if len(j) > 0:
                target = target[:j[0]]

            return reward_function(output, target)

        def compute_rewards(outputs, targets):
            return np.array([compute_reward(output, target) for output, target in zip(outputs, targets)])

        rewards = compute_rewards(samples, targets[0][:,1:])
        rewards = np.stack([rewards] * samples.shape[1], axis=1)

        input_feed[self.outputs[0]] = outputs[0]
        input_feed[self.samples] = samples
        input_feed[self.rewards] = rewards

        output_feed = {'loss': self.reinforce_loss, 'baseline_loss': self.baseline_loss}
        if update_model:
            output_feed['update'] = self.update_ops.reinforce[1] if use_sgd else self.update_ops.reinforce[0]
        if self.use_baseline and update_baseline:
            output_feed['baseline_update'] = self.update_ops.baseline[0]  # FIXME

        if align:
            output_feed['weights'] = self.attention_weights

        res = session.run(output_feed, input_feed)
        return namedtuple('output', 'loss weights baseline_loss')(res['loss'], res.get('weights'),
                                                                  res.get('baseline_loss'))

    def step(self, session, data, update_model=True, align=False, use_sgd=False, **kwargs):
        if self.dropout is not None:
            session.run(self.dropout_on)

        encoder_inputs, targets, input_length = self.get_batch(data)
        input_feed = {self.targets: targets}

        for i in range(len(self.encoders)):
            input_feed[self.encoder_inputs[i]] = encoder_inputs[i]
            input_feed[self.encoder_input_length[i]] = input_length[i]

        output_feed = {'loss': self.xent_loss}
        if update_model:
            output_feed['update'] = self.update_ops.xent[1] if use_sgd else self.update_ops.xent[0]
        if align:
            output_feed['weights'] = self.attention_weights

        res = session.run(output_feed, input_feed)
        return namedtuple('output', 'loss weights')(res['loss'], res.get('weights'))

    def greedy_decoding(self, session, token_ids):
        if self.dropout is not None:
            session.run(self.dropout_off)

        data = [
            ids + [[] for _ in self.decoders] if len(ids) == len(self.encoders) else ids
            for ids in token_ids
        ]

        batch = self.get_batch(data, decoding=True)
        encoder_inputs, targets, input_length = batch

        input_feed = {self.targets: targets, self.feed_previous: 1.0}

        for i in range(len(self.encoders)):
            input_feed[self.encoder_inputs[i]] = encoder_inputs[i]
            input_feed[self.encoder_input_length[i]] = input_length[i]

        outputs = session.run(self.outputs, input_feed)
        return [np.argmax(outputs_, axis=2) for outputs_ in outputs]

    def beam_search_decoding(self, session, token_ids, beam_size, early_stopping=True):
        if not isinstance(session, list):
            session = [session]

        if self.dropout is not None:
            for session_ in session:
                session_.run(self.dropout_off)

        data = [token_ids + [[]]]
        encoder_inputs, targets, input_length = self.get_batch(data, decoding=True)
        targets = targets[0]  # multi-decoder not supported
        input_feed = {}

        for i in range(len(self.encoders)):
            input_feed[self.encoder_inputs[i]] = encoder_inputs[i]
            input_feed[self.encoder_input_length[i]] = input_length[i]

        output_feed = [self.encoder_state] + self.attention_states
        res = [session_.run(output_feed, input_feed) for session_ in session]
        state, attn_states = list(zip(*[(res_[0], res_[1:]) for res_ in res]))

        targets = targets[:,0]  # BOS symbol

        finished_hypotheses = []
        finished_scores = []

        hypotheses = [[]]
        scores = np.zeros([1], dtype=np.float32)

        beam_data = None

        for k in range(self.max_output_len[0]):
            batch_size = targets.shape[0]
            targets = np.reshape(targets, [batch_size, 1])
            targets = np.concatenate([targets, np.ones(targets.shape) * utils.EOS_ID], axis=1)

            input_feed = [{self.targets[0]: targets} for _ in session]

            if beam_data is not None:
                for feed, data_ in zip(input_feed, beam_data):
                    feed[self.beam_tensors.data] = data_

            for feed, attn_states_ in zip(input_feed, attn_states):
                for i in range(len(self.encoders)):
                    feed[self.encoder_inputs[i]] = encoder_inputs[i]
                    feed[self.encoder_input_length[i]] = input_length[i]
                    feed[self.attention_states[i]] = attn_states_[i].repeat(batch_size, axis=0)

            output_feed = [self.beam_tensors.new_data, self.beam_output]

            res = [session_.run(output_feed, input_feed_) for session_, input_feed_ in zip(session, input_feed)]
            beam_data, proba = list(zip(*res))

            proba = [np.maximum(proba_, 1e-30) for proba_ in proba]

            scores_ = scores[:, None] - np.average([np.log(proba_) for proba_ in proba], axis=0)
            scores_ = scores_.flatten()
            flat_ids = np.argsort(scores_)

            token_ids_ = flat_ids % self.decoders[0].vocab_size
            hyp_ids = flat_ids // self.decoders[0].vocab_size

            new_hypotheses = []
            new_scores = []
            new_data = [[] for _ in session]
            new_input = []

            for flat_id, hyp_id, token_id in zip(flat_ids, hyp_ids, token_ids_):
                if len(new_hypotheses) == beam_size:
                    break

                hypothesis = hypotheses[hyp_id] + [token_id]
                score = scores_[flat_id]

                if token_id == utils.EOS_ID:
                    # hypothesis is finished, it is thus unnecessary to keep expanding it
                    finished_hypotheses.append(hypothesis)
                    finished_scores.append(score)

                    # early stop: number of possible hypotheses is reduced by one
                    if early_stopping:
                        beam_size -= 1
                else:
                    new_hypotheses.append(hypothesis)

                    for session_id, data_, in enumerate(beam_data):
                        new_data[session_id].append(data_[hyp_id])

                    new_scores.append(score)
                    new_input.append(token_id)

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
        inputs = [[] for _ in self.encoders]
        targets = [[] for _ in self.decoders]
        input_length = [[] for _ in self.encoders]

        # maximum input length of each encoder in this batch
        max_input_len = [max(len(data_[i]) for data_ in data) for i in range(len(self.encoders))]

        if self.max_input_len is not None:
            max_input_len = [min(len_, max_len) for len_, max_len in zip(max_input_len, self.max_input_len)]

        # maximum output length in this batch
        if decoding:
            max_output_len = self.max_output_len
        else:
            max_output_len = [max(len(data_[i]) for data_ in data)
                              for i in range(len(self.encoders), len(self.encoders) + len(self.decoders))]
            if self.max_output_len is not None:
                max_output_len = [min(len_, max_len) for len_, max_len in zip(max_output_len, self.max_output_len)]

        for sentences in data:
            src_sentences = sentences[:len(self.encoders)]
            trg_sentences = sentences[len(self.encoders):]

            for i, (encoder, src_sentence) in enumerate(zip(self.encoders, src_sentences)):
                src_sentence = src_sentence[:max_input_len[i]]
                pad_symbol = np.zeros(encoder.embedding_size, dtype=np.float32) if encoder.binary else utils.EOS_ID
                # pad sequences so that all sequences in the same batch have the same length

                eos = 0 if encoder.binary else 1   # end of sentence marker for non-binary input
                encoder_pad = [pad_symbol] * (eos + max_input_len[i] - len(src_sentence))
                inputs[i].append(src_sentence + encoder_pad)
                input_length[i].append(len(src_sentence) + eos)
                # inputs[i].append(src_sentence[::-1] + encoder_pad)  # reverse sequence

            for i in range(len(targets)):
                if decoding:
                    targets[i].append([utils.BOS_ID] * self.max_output_len[i] + [utils.EOS_ID])
                else:
                    trg_sentence = trg_sentences[i][:max_output_len[i]]
                    decoder_pad_size = max_output_len[i] - len(trg_sentence) + 1
                    trg_sentence = [utils.BOS_ID] + trg_sentence + [utils.EOS_ID] * decoder_pad_size
                    targets[i].append(trg_sentence)

        # convert lists to numpy arrays
        inputs = [np.array(inputs_, dtype=np.float32 if encoder.binary else np.int32)
                  for encoder, inputs_ in zip(self.encoders, inputs)]
        # starts with BOS and ends with EOS
        targets = [np.array(targets_, dtype=np.int32) for targets_ in targets]
        input_length = [np.array(input_length_, dtype=np.int32) for input_length_ in input_length]

        return inputs, targets, input_length
