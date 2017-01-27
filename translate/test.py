import math
import tensorflow as tf
import numpy as np
import os
import shutil
from collections import Counter
from translate.utils import sentence_to_token_ids, initialize_vocabulary, EOS_ID

load = True

sess = tf.Session()

if load:
    new_saver = tf.train.import_meta_graph('test/model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./test'))

    c = tf.get_default_graph().get_tensor_by_name('c:0')
else:
    shutil.rmtree('test', ignore_errors=True)
    os.mkdir('test')

    # create graph
    a = tf.get_variable('a', [10, 20])
    b = tf.get_variable('b', [10, 20])
    c = tf.identity(a + b, 'c')
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.save(sess, 'test/model')
