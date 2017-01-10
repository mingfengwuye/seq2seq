import tensorflow as tf
import numpy as np

indices = tf.constant([1, 3, 2])

indices = tf.stack([tf.range(tf.shape(indices)[0]), indices], axis=1)
values = tf.get_variable('w', shape=[3, 4])

x = tf.gather_nd(values ** 2, indices)

gradients = tf.gradients(x, values)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

a = tf.constant(np.arange(1,13).astype(np.float32), shape=[2, 2, 3])
b = tf.constant(np.arange(13,25).astype(np.float32), shape=[2, 3, 2])
c = sess.run(tf.matmul(a, b))

# a = tf.get_variable('a', shape=[2, 5, 3])
# b = tf.get_variable('b', shape=[3, 1])

# c = tf.matmul(a, b)

pass