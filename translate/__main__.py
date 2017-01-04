"""Script for training translation models and decoding from them

See the following papers for more information on neural translation models
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
import os
import sys
import logging
import argparse
import subprocess
import tensorflow as tf
import yaml
import shutil

from pprint import pformat
from operator import itemgetter
from translate import utils
from translate.multitask_model import MultiTaskModel

parser = argparse.ArgumentParser()
parser.add_argument('config', help='load a configuration file in the YAML format')
parser.add_argument('-v', '--verbose', help='verbose mode', action='store_true')
parser.add_argument('--reset', help="reset model (don't load any checkpoint)", action='store_true')
parser.add_argument('--reset-learning-rate', help='reset learning rate', action='store_true')
parser.add_argument('--learning-rate', type=float, help='custom learning rate (triggers `reset-learning-rate`)')
parser.add_argument('--purge', help='remove previous model files', action='store_true')

# Available actions (exclusive)
parser.add_argument('--decode', help='translate this corpus (one filename for each encoder)', nargs='*')
parser.add_argument('--align', help='translate and show alignments by the attention mechanism', nargs=2)
parser.add_argument('--eval', help='compute BLEU score on this corpus (source files and target file)', nargs='+')
parser.add_argument('--train', help='train an NMT model', action='store_true')

# TensorFlow configuration
parser.add_argument('--gpu-id', type=int, help='index of the GPU where to run the computation')
parser.add_argument('--no-gpu', action='store_true', help='run on CPU')

# Decoding options (to avoid having to edit the config file)
parser.add_argument('--beam-size', type=int)
parser.add_argument('--ensemble', action='store_const', const=True)
parser.add_argument('--lm-file')
parser.add_argument('--checkpoints', nargs='+')
parser.add_argument('--lm-weight', type=float)
parser.add_argument('--len-normalization', type=float)
parser.add_argument('--output')
parser.add_argument('--max-steps', type=int)
parser.add_argument('--remove-unk', action='store_const', const=True)
parser.add_argument('--wav-files', nargs='*')

"""
Benchmarks:
- replicate speech recognition results
- replicate the experiments of the WMT paper on neural post-editing

Scores on English->French WMT14 (ntst14)
- Bahdanau et al.:  28.45 (shallow attention-based MT with max-out layer, 30k vocabulary)
- Baseline (Moses): 33.30
- Sutskever et al.: 30.59 (deep LSTMs with large vocab, huge model but without attention)
- Jean et al.:      29.97 (same model as Bahdanau et al. + tuning while freezing embeddings)
                    32.68 (with larger vocabulary + sampled softmax)

Sutskever and Jean use a beam-size of 12, while Bahdanau uses beam-search with unspecified beam-size (might explain
the difference in score in addition to tuning)


TODO:
- pervasive dropout (dropout in the recurrent connections)
- reading files as a stream when decoding (useful for large files)
- symbolic beam-search
- possibility to build an encoder with 1 bi-directional layer, and several uni-directional layers
- pre-load data on GPU for small datasets
- mixture of Adam and SGD training
- decay learning rate after a certain number of epochs
- possibility to run model on several GPUs
- copy vocab and config to model dir
"""


def main(args=None):
    args = parser.parse_args(args)

    # read config file and default config
    with open('config/default.yaml') as f:
        default_config = utils.AttrDict(yaml.safe_load(f))

    with open(args.config) as f:
        config = utils.AttrDict(yaml.safe_load(f))
        
        if args.learning_rate is not None:
            args.reset_learning_rate = True
        
        # command-line parameters have higher precedence than config file
        for k, v in vars(args).items():
            if v is not None:
                config[k] = v

        # set default values for parameters that are not defined
        for k, v in default_config.items():
            config.setdefault(k, v)

    # enforce parameter constraints
    assert config.steps_per_eval % config.steps_per_checkpoint == 0, (
        'steps-per-eval should be a multiple of steps-per-checkpoint')
    assert args.decode is not None or args.eval or args.train or args.align, (
        'you need to specify at least one action (decode, eval, align, or train)')

    if args.purge:
        utils.log('deleting previous model')
        shutil.rmtree(config.model_dir, ignore_errors=True)

    logging_level = logging.DEBUG if args.verbose else logging.INFO
    # always log to stdout in decoding and eval modes (to avoid overwriting precious train logs)
    logger = utils.create_logger(config.log_file if args.train else None)
    logger.setLevel(logging_level)

    utils.log(' '.join(sys.argv))  # print command line
    try:  # print git hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        utils.log('commit hash {}'.format(commit_hash))
    except:
        pass

    # list of encoder and decoder parameter names (each encoder and decoder can have a different value
    # for those parameters)
    model_parameters = [
        'cell_size', 'layers', 'vocab_size', 'embedding_size', 'attention_filters', 'attention_filter_length',
        'use_lstm', 'time_pooling', 'attention_window_size', 'dynamic', 'binary', 'character_level', 'bidir',
        'load_embeddings', 'pooling_avg', 'swap_memory', 'parallel_iterations', 'input_layers',
        'residual_connections', 'attn_size'
    ]
    # TODO: independent model dir for each task
    task_parameters = [
        'data_dir', 'train_prefix', 'dev_prefix', 'vocab_prefix', 'ratio', 'lm_file', 'learning_rate',
        'learning_rate_decay_factor', 'max_input_len', 'max_output_len', 'encoders', 'decoder'
    ]

    # in case no task is defined (standard mono-task settings), define a "main" task
    config.setdefault(
        'tasks', [{'encoders': config.encoders, 'decoder': config.decoder, 'name': 'main', 'ratio': 1.0}]
    )
    config.tasks = [utils.AttrDict(task) for task in config.tasks]

    for task in config.tasks:
        for parameter in task_parameters:
            task.setdefault(parameter, config.get(parameter))

        if isinstance(task.dev_prefix, str):  # for back-compatibility with old config files
            task.dev_prefix = [task.dev_prefix]

        # convert dicts to AttrDicts for convenience
        task.encoders = [utils.AttrDict(encoder) for encoder in task.encoders]
        task.decoder = utils.AttrDict(task.decoder)

        for encoder_or_decoder in task.encoders + [task.decoder]:
            # move parameters all the way up from base level to encoder/decoder level:
            # default values for encoder/decoder parameters can be defined at the task level and base level
            # default values for tasks can be defined at the base level
            for parameter in model_parameters:
                if parameter in encoder_or_decoder:
                    continue
                elif parameter in task:
                    encoder_or_decoder[parameter] = task[parameter]
                else:
                    encoder_or_decoder[parameter] = config.get(parameter)

    # log parameters
    utils.log('program arguments')
    for k, v in sorted(config.items(), key=itemgetter(0)):
        if k == 'tasks':
            utils.log('  {:<20}\n{}'.format(k, pformat(v)))
        elif k not in model_parameters and k not in task_parameters:
            utils.log('  {:<20} {}'.format(k, pformat(v)))

    device = None
    if args.no_gpu:
        device = '/cpu:0'
    elif args.gpu_id is not None:
        device = '/gpu:{}'.format(args.gpu_id)

    utils.log('creating model')
    utils.log('using device: {}'.format(device))

    with tf.device(device):
        checkpoint_dir = os.path.join(config.model_dir, 'checkpoints')
        # All parameters except recurrent connexions and attention parameters are initialized with this.
        # Recurrent connexions are initialized with orthogonal matrices, and the parameters of the attention model
        # with a standard deviation of 0.001
        if config.weight_scale:
            initializer = tf.random_normal_initializer(stddev=config.weight_scale)
        else:
            initializer = None

        tf.get_variable_scope().set_initializer(initializer)
        decode_only = args.decode is not None or args.eval or args.align  # exempt from creating gradient ops
        model = MultiTaskModel(name='main', checkpoint_dir=checkpoint_dir, decode_only=decode_only, **config)

    utils.log('model parameters ({})'.format(len(tf.global_variables())))
    parameter_count = 0
    for var in tf.global_variables():
        utils.log('  {} {}'.format(var.name, var.get_shape()))

        v = 1
        for d in var.get_shape():
            v *= d.value
        parameter_count += v
    utils.log('number of parameters: {}'.format(parameter_count))

    tf_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = config.allow_growth
    tf_config.gpu_options.per_process_gpu_memory_fraction = config.mem_fraction

    with tf.Session(config=tf_config) as sess:
        best_checkpoint = os.path.join(checkpoint_dir, 'best')

        if config.ensemble and (args.eval or args.decode is not None):
            # create one session for each model in the ensemble
            sess = [tf.Session() for _ in config.checkpoints]
            for sess_, checkpoint in zip(sess, config.checkpoints):
                model.initialize(sess_, [checkpoint], reset=True)
        elif (not config.checkpoints and (args.eval or args.decode is not None or args.align)
              and os.path.isfile(best_checkpoint)):
            # in decoding and evaluation mode, unless specified otherwise (by `checkpoints` or `reset` parameters,
            # try to load the best checkpoint)
            model.initialize(sess, [best_checkpoint], reset=True)
        else:
            # loads last checkpoint, unless `reset` is true
            model.initialize(sess, config.checkpoints, reset=args.reset, reset_learning_rate=args.reset_learning_rate)

        # Inspect variables:
        # tf.get_variable_scope().reuse_variables()
        # import pdb; pdb.set_trace()

        if args.decode is not None:
            model.decode(sess, **config)
        elif args.eval:
            model.evaluate(sess, on_dev=False, **config)
        elif args.align:
            model.align(sess, **config)
        elif args.train:
            eval_output = os.path.join(config.model_dir, 'eval')
            try:
                model.train(sess, eval_output=eval_output, **config)
            except KeyboardInterrupt:
                utils.log('exiting...')
                model.save(sess)
                sys.exit()


if __name__ == '__main__':
    main()
