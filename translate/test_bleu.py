import tensorflow as tf
import numpy as np

from time import time
from translate.utils import sentence_bleu as py_bleu
from translate.decoders import batch_bleu
from translate.utils import initialize_vocabulary, sentence_to_token_ids, EOS_ID

vocab_filename = 'data/btec/vocab.en'
hyp_filename = 'models/btec_xent.bak/eval.main.4000'
ref_filename = 'data/btec/dev.500.en'

vocab = initialize_vocabulary(vocab_filename).vocab

sess = tf.InteractiveSession()

def pad(s, l):
    return s + [EOS_ID] * (l - len(s))


with open(hyp_filename) as hyp_file, open(ref_filename) as ref_file:
    hyps = list(hyp_file)
    refs = list(ref_file)

    hyps = [sentence_to_token_ids(hyp, vocab) for hyp in hyps]
    refs = [sentence_to_token_ids(ref, vocab) for ref in refs]

    max_hyp_len = max(map(len, hyps))
    max_ref_len = max(map(len, refs))

    padded_hyps = [pad(hyp, max_hyp_len) for hyp in hyps]
    padded_refs = [pad(ref, max_ref_len) for ref in refs]

    with tf.device('/cpu:0'):
        hyp_tensor = tf.constant(padded_hyps)
        ref_tensor = tf.constant(padded_refs)
        bleu_fun = batch_bleu(hyp_tensor, ref_tensor)

    time1 = time()
    bleus1 = [py_bleu(hyp, ref) for hyp, ref in zip(hyps, refs)]
    time1 = time() - time1

    time2 = time()
    bleus2 = bleu_fun.eval()
    time2 = time() - time2

    print('py:', time1)
    print('tf:', time2)

    assert np.isclose(bleus1, bleus2).all()