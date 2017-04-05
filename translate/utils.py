import os
import sys
import re
import numpy as np
import logging
import struct
import random
import math
import wave
import shutil

from collections import namedtuple
from contextlib import contextmanager

# special vocabulary symbols

_BOS = '<S>'
_EOS = '</S>'
_UNK = '<UNK>'
_KEEP = '<KEEP>'
_DEL = '<DEL>'
_INS = '<INS>'
_SUB = '<SUB>'
_NONE = '<NONE>'

_START_VOCAB = [_BOS, _EOS, _UNK, _KEEP, _DEL, _INS, _SUB, _NONE]

BOS_ID = 0
EOS_ID = 1
UNK_ID = 2
KEEP_ID = 3
DEL_ID = 4
INS_ID = 5
SUB_ID = 6
NONE_ID = 7

class FinishedTrainingException(Exception):
    def __init__(self):
        debug('finished training')


@contextmanager
def open_files(names, mode='r'):
    """ Safely open a list of files in a context manager.
    Example:
    >>> with open_files(['foo.txt', 'bar.csv']) as (f1, f2):
    ...   pass
    """

    files = []
    try:
        for name_ in names:
            files.append(open(name_, mode=mode))
        yield files
    finally:
        for file_ in files:
            file_.close()


class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    Example:
    >>> d = AttrDict(x=1, y=2)
    >>> d.x
    1
    >>> d.y = 3
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self  # dark magic

    def __getattr__(self, item):
        return self.__dict__.get(item)

def reverse_edits(source, edits, fix=True, strict=False):
    src_words = source.split()
    target = []

    i = 0

    consistent = True

    for edit in edits.split():
        if strict and not consistent:
            break

        if edit in (_DEL, _KEEP, _SUB) or edit.startswith(_SUB + '_'):
            if i >= len(src_words):
                consistent = False
                continue

            if edit == _KEEP:
                target.append(src_words[i])
            elif edit == _SUB:
                target.append(edit)
            elif edit.startswith(_SUB + '_'):
                target.append(edit[len(_SUB + '_'):])

            i += 1
        elif edit.startswith(_INS + '_'):
            target.append(edit[len(_INS + '_'):])
        else:
            target.append(edit)

    if fix:
        target += src_words[i:]

    return ' '.join(target)


def initialize_vocabulary(vocabulary_path):
    """
    Initialize vocabulary from file.

    We assume the vocabulary is stored one-item-per-line, so a file:
      dog
      cat
    will result in a vocabulary {'dog': 0, 'cat': 1}, and a reversed vocabulary ['dog', 'cat'].

    :param vocabulary_path: path to the file containing the vocabulary.
    :return:
      the vocabulary (a dictionary mapping string to integers), and
      the reversed vocabulary (a list, which reverses the vocabulary mapping).
    """
    if os.path.exists(vocabulary_path):
        rev_vocab = []
        with open(vocabulary_path) as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.rstrip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return namedtuple('vocab', 'vocab reverse')(vocab, rev_vocab)
    else:
        raise ValueError("vocabulary file %s not found", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, character_level=False):
    """
    Convert a string to list of integers representing token-ids.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    :param sentence: a string, the sentence to convert to token-ids
    :param vocabulary: a dictionary mapping tokens to integers
    :param character_level: treat sentence as a string of characters, and
        not as a string of words
    :return: a list of integers, the token-ids for the sentence.
    """
    sentence = sentence.rstrip('\n') if character_level else sentence.split()
    return [vocabulary.get(w, UNK_ID) for w in sentence]


def get_filenames(data_dir, model_dir, extensions, train_prefix, dev_prefix, vocab_prefix, **kwargs):
    """
    Get a bunch of file prefixes and extensions, and output the list of filenames to be used
    by the model.

    :param data_dir: directory where all the the data is stored
    :param extensions: list of file extensions, in the right order (last extension is always the target)
    :param train_prefix: name of the training corpus (usually 'train')
    :param dev_prefix: name of the dev corpus (usually 'dev')
    :param vocab_prefix: prefix of the vocab files (usually 'vocab')
    :param kwargs: optional contains an additional 'decode', 'eval' or 'align' parameter
    :return: namedtuple containing the filenames
    """
    train_path = os.path.join(data_dir, train_prefix)
    dev_path = [os.path.join(data_dir, prefix) for prefix in dev_prefix]

    train = ['{}.{}'.format(train_path, ext) for ext in extensions]
    dev = [['{}.{}'.format(path, ext) for ext in extensions] for path in dev_path]

    vocab_path = os.path.join(data_dir, vocab_prefix)
    vocab_src = ['{}.{}'.format(vocab_path, ext) for ext in extensions]

    vocab_path = os.path.join(model_dir, 'data', 'vocab')
    vocab = ['{}.{}'.format(vocab_path, ext) for ext in extensions]
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)

    for src, dest in zip(vocab_src, vocab):
        if not os.path.exists(dest):
            debug('copying vocab to {}'.format(dest))
            shutil.copy(src, dest)

    test = kwargs.get('decode')  # empty list means we decode from standard input
    if test is None:
        test = test or kwargs.get('eval')
        test = test or kwargs.get('align')

    filenames = namedtuple('filenames', ['train', 'dev', 'test', 'vocab'])
    return filenames(train, dev, test, vocab)


def read_dataset(paths, extensions, vocabs, max_size=None, character_level=None, sort_by_length=False,
                 max_seq_len=None):
    data_set = []

    line_reader = read_lines(paths, extensions)
    character_level = character_level or [False] * len(extensions)

    for counter, inputs in enumerate(line_reader, 1):
        if max_size and counter > max_size:
            break
        if counter % 100000 == 0:
            log("  reading data line {}".format(counter))

        inputs = [
            sentence_to_token_ids(input_, vocab.vocab, character_level=char_level)
            if vocab is not None and isinstance(input_, str)
            else input_
            for input_, vocab, ext, char_level in zip(inputs, vocabs, extensions, character_level)
        ]

        if not all(inputs):  # skip empty inputs
            continue
        # skip lines that are too long
        if max_seq_len and any(len(inputs_) > max_len for inputs_, max_len in zip(inputs, max_seq_len)):
            continue

        data_set.append(inputs)  # TODO: filter too long

    debug('files: {}'.format(' '.join(paths)))
    debug('size: {}'.format(len(data_set)))

    if sort_by_length:
        data_set.sort(key=lambda lines: list(map(len, lines)))

    return data_set


def random_batch_iterator(data, batch_size):
    """
    The most basic form of batch iterator.

    :param data: the dataset to segment into batches
    :param batch_size: the size of a batch
    :return: an iterator which yields random batches (indefinitely)
    """
    while True:
        yield random.sample(data, batch_size)


def cycling_batch_iterator(data, batch_size, shuffle=True, allow_smaller=True):
    """
    Indefinitely cycle through a dataset and yield batches (the dataset is shuffled
    at each new epoch)

    :param data: the dataset to segment into batches
    :param batch_size: the size of a batch
    :return: an iterator which yields batches (indefinitely)
    """
    while True:
        if shuffle:
            random.shuffle(data)

        batch_count = len(data) // batch_size

        if allow_smaller and batch_count * batch_size < len(data):
            batch_count += 1

        for i in range(batch_count):
            yield data[i * batch_size:(i + 1) * batch_size]


def read_ahead_batch_iterator(data, batch_size, read_ahead=10, shuffle=True, allow_smaller=True,
                              mode='standard', **kwargs):
    """
    Same iterator as `cycling_batch_iterator`, except that it reads a number of batches
    at once, and sorts their content according to their size.

    This is useful for training, where all the sequences in one batch need to be padded
     to the same length as the longest sequence in the batch.

    :param data: the dataset to segment into batches
    :param batch_size: the size of a batch
    :param read_ahead: number of batches to read ahead of time and sort (larger numbers
      mean faster training, but less random behavior)
    :return: an iterator which yields batches (indefinitely)
    """
    if mode == 'random':
        iterator = random_batch_iterator(data, batch_size)
    else:
        iterator = cycling_batch_iterator(data, batch_size, shuffle=shuffle, allow_smaller=allow_smaller)

    if read_ahead <= 1:
        while True:
            yield next(iterator)

    while True:
        batches = [next(iterator) for _ in range(read_ahead)]
        data_ = sorted(sum(batches, []), key=lambda lines: len(lines[-1]))
        batches = [data_[i * batch_size:(i + 1) * batch_size] for i in range(read_ahead)]
        if shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch


def get_batches(data, batch_size, batches=0, allow_smaller=True):
    """
    Segment `data` into a given number of fixed-size batches. The dataset is automatically shuffled.

    This function is for smaller datasets, when you need access to the entire dataset at once (e.g. dev set).
    For larger (training) datasets, where you may want to lazily iterate over batches
    and cycle several times through the entire dataset, prefer batch iterators
    (such as `cycling_batch_iterator`).

    :param data: the dataset to segment into batches (a list of data points)
    :param batch_size: the size of a batch
    :param batches: number of batches to return (0 for the largest possible number)
    :param allow_smaller: allow the last batch to be smaller
    :return: a list of batches (which are lists of `batch_size` data points)
    """
    if not allow_smaller:
        max_batches = len(data) // batch_size
    else:
        max_batches = int(math.ceil(len(data) / batch_size))

    if batches < 1 or batches > max_batches:
        batches = max_batches

    random.shuffle(data)
    batches = [data[i * batch_size:(i + 1) * batch_size] for i in range(batches)]
    return batches


def read_lines(paths, extensions):
    if not paths:  # read from stdin (only works with one encoder with text input)
        assert len(extensions) == 1
        paths = [None]

    iterators = [
        sys.stdin if filename is None else open(filename)
        for ext, filename in zip(extensions, paths,)
    ]

    return zip(*iterators)


def create_logger(log_file=None):
    """
    Initialize global logger and return it.

    :param log_file: log to this file, or to standard output if None
    :return: created logger
    """
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S')
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)
        logger = logging.getLogger(__name__)
        logger.addHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    return logger


def log(msg, level=logging.INFO):
    logging.getLogger(__name__).log(level, msg)


def debug(msg): log(msg, level=logging.DEBUG)


def warn(msg): log(msg, level=logging.WARN)


def heatmap(xlabels=None, ylabels=None, weights=None, output_file=None):
    """
    Draw a heatmap showing the alignment between two sequences.

    :param xlabels: input words
    :param ylabels: output words
    :param weights: numpy array of shape (len(xlabels), len(ylabels))
    :param output_file: write the figure to this file, or show it into a window if None
    """
    from matplotlib import pyplot as plt

    xlabels = xlabels or []
    ylabels = ylabels or []

    fig, ax = plt.subplots()

    plt.autoscale(enable=True, axis='x', tight=True)
    ax.pcolor(weights, cmap=plt.cm.Greys)
    ax.set_frame_on(False)
    # plt.colorbar(mappable=heatmap_)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(weights.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(weights.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(xlabels, minor=False)
    ax.set_yticklabels(ylabels, minor=False)
    ax.tick_params(axis='both', which='both', length=0)

    plt.xticks(rotation=90, fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    # ax.set_aspect('equal')
    ax.grid(True)

    xsize = max(2.0 + len(xlabels) / 3, 8.0)
    ysize = max(2.0 + len(ylabels) / 3, 8.0)
    fig.set_size_inches(xsize, ysize, forward=True)

    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file)
