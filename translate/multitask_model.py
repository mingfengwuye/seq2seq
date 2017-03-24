import time
import math
import os
import numpy as np
from translate import utils
from translate.translation_model import TranslationModel, BaseTranslationModel


class MultiTaskModel(BaseTranslationModel):
    def __init__(self, name, tasks, checkpoint_dir, model_dir, keep_best=1, main_task=None, **kwargs):
        """
        Proxy for several translation models that are trained jointly
        This class presents the same interface as TranslationModel
        """
        super(MultiTaskModel, self).__init__(name, checkpoint_dir, keep_best, **kwargs)

        self.models = []
        self.ratios = []

        assert len(set(task.name for task in tasks)) == len(tasks), 'error: task names should be unique'
        assert not any(task.name is None for task in tasks), 'error: undefined task name'

        for task in tasks:
            self.checkpoint_dir = checkpoint_dir
            # merging both dictionaries (task parameters have a higher precedence)
            kwargs_ = dict(**kwargs)
            kwargs_.update(task)

            dest_vocab_path = os.path.join(model_dir, 'data', '{}.vocab'.format(task.name))
            model = TranslationModel(checkpoint_dir=None, keep_best=keep_best, dest_vocab_path=dest_vocab_path,
                                     **kwargs_)

            self.models.append(model)
            self.ratios.append(task.ratio if task.ratio is not None else 1)

        self.ratios = [ratio / sum(self.ratios) for ratio in self.ratios]  # unit normalization

        self.main_task = main_task
        self.global_step = 0  # steps of all tasks combined


    def train(self, sess, beam_size, steps_per_checkpoint, steps_per_eval=None, eval_output=None, max_steps=0,
              max_epochs=0, eval_burn_in=0, decay_if_no_progress=5, decay_after_n_epoch=None, decay_every_n_epoch=None,
              sgd_after_n_epoch=None, loss_function='xent', baseline_steps=0, reinforce_baseline=True,
              reward_function=None, **kwargs):
        utils.log('reading training and development data')

        self.global_step = 0
        for model in self.models:
            model.read_data(**kwargs)
            # those parameters are used to track the progress of each task
            model.loss, model.time, model.steps = 0, 0, 0
            model.baseline_loss = 0
            model.previous_losses = []
            global_step = model.global_step.eval(sess)
            model.epoch = model.batch_size * global_step // model.train_size
            model.last_decay = global_step

            for _ in range(global_step):   # read all the data up to this step
                next(model.batch_iterator)

            self.global_step += global_step

        # pre-train baseline
        if loss_function == 'reinforce' and baseline_steps > 0 and reinforce_baseline:
            utils.log('pre-training baseline')
            for model in self.models:
                baseline_loss = 0
                for step in range(1, baseline_steps + 1):
                    baseline_loss += model.baseline_step(sess, reward_function=reward_function)

                    if step % steps_per_checkpoint == 0:
                        loss = baseline_loss / steps_per_checkpoint
                        baseline_loss = 0
                        utils.log('{} step {} baseline loss {:.4f}'.format(model.name, step, loss))

        utils.log('starting training')
        while True:
            i = np.random.choice(len(self.models), 1, p=self.ratios)[0]
            model = self.models[i]

            start_time = time.time()
            res = model.train_step(sess, loss_function=loss_function, reward_function=reward_function)
            model.loss += res.loss

            if loss_function == 'reinforce':
                model.baseline_loss += res.baseline_loss

            model.time += time.time() - start_time
            model.steps += 1
            self.global_step += 1
            model_global_step = model.global_step.eval(sess)

            epoch = model.batch_size * model_global_step / model.train_size
            model.epoch = int(epoch) + 1

            if decay_after_n_epoch is not None and epoch >= decay_after_n_epoch:
                if decay_every_n_epoch is not None and (model.batch_size * (model_global_step - model.last_decay)
                                                            >= decay_every_n_epoch * model.train_size):
                    sess.run(model.learning_rate_decay_op)
                    utils.debug('  decaying learning rate to: {:.4f}'.format(model.learning_rate.eval()))
                    model.last_decay = model_global_step

            if sgd_after_n_epoch is not None and epoch >= sgd_after_n_epoch:
                if not model.use_sgd:
                    utils.debug('  epoch {}, starting to use SGD'.format(model.epoch))
                    model.use_sgd = True

            if steps_per_checkpoint and self.global_step % steps_per_checkpoint == 0:
                for model_ in self.models:
                    if model_.steps == 0:
                        continue

                    loss_ = model_.loss / model_.steps
                    step_time_ = model_.time / model_.steps

                    if loss_function == 'reinforce':
                        baseline_loss_ = ' baseline loss {:.4f}'.format(model_.baseline_loss / model_.steps)
                        model_.baseline_loss = 0
                    else:
                        baseline_loss_ = ''

                    utils.log('{} step {} epoch {} learning rate {:.4f} step-time {:.4f}{} loss {:.4f}'.format(
                        model_.name, model_.global_step.eval(sess), model.epoch, model_.learning_rate.eval(),
                        step_time_, baseline_loss_, loss_))
                    
                    if decay_if_no_progress and len(model_.previous_losses) >= decay_if_no_progress:
                        if loss_ >= max(model_.previous_losses[:decay_if_no_progress]):
                            sess.run(model_.learning_rate_decay_op)

                    model_.previous_losses.append(loss_)
                    model_.loss, model_.time, model_.steps = 0, 0, 0
                    model_.eval_step(sess)

                self.save(sess)

            if steps_per_eval and self.global_step % steps_per_eval == 0 and 0 <= eval_burn_in <= self.global_step:
                score = 0

                for ratio, model_ in zip(self.ratios, self.models):
                    step = model_.global_step.eval(sess)

                    if eval_output is None:
                        output = None
                    else:
                        os.makedirs(eval_output, exist_ok=True)

                        # if there are several dev files, we define several output files
                        output = [
                            os.path.join(eval_output, '{}.{}.{}'.format(model_.name, prefix, step))
                            for prefix in model_.dev_prefix
                        ]

                    # kwargs_ = {**kwargs, 'output': output}
                    kwargs_ = dict(kwargs)
                    kwargs_['output'] = output
                    scores_ = model_.evaluate(sess, beam_size, on_dev=True, **kwargs_)
                    score_ = scores_[0]  # in case there are several dev files, only the first one counts

                    # if there is a main task, pick best checkpoint according to its score
                    # otherwise use the average score across tasks
                    if self.main_task is None:
                        score += ratio * score_
                    elif model_.name == self.main_task:
                        score = score_

                self.manage_best_checkpoints(self.global_step, score)

            if 0 < max_steps <= self.global_step or 0 < max_epochs <= epoch:
                raise utils.FinishedTrainingException

    def decode(self, *args, **kwargs):
        if self.main_task is not None:
            model = next(model for model in self.models if model.name == self.main_task)
        else:
            model = self.models[0]
        return model.decode(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        if self.main_task is not None:
            model = next(model for model in self.models if model.name == self.main_task)
        else:
            model = self.models[0]
        return model.evaluate(*args, **kwargs)

    def align(self, *args, **kwargs):
        if self.main_task is not None:
            model = next(model for model in self.models if model.name == self.main_task)
        else:
            model = self.models[0]
        return model.align(*args, **kwargs)
