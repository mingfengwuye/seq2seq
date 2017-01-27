import tensorflow as tf
from translate import utils

sess = tf.InteractiveSession()

hyp = tf.constant([31, 60, 17, 9, 184, 4], tf.int64)
ref = tf.constant([31, 60, 17, 38, 184, 4], tf.int64)

def truncate(s):
    indices = tf.squeeze(tf.where(tf.equal(s, utils.EOS_ID)), 1)
    indices = tf.concat(0, [tf.cast(indices, tf.int32), tf.shape(s)])
    index = indices[0]
    return s[:index]

hyp = tf.cast(truncate(hyp), tf.int64) + 1
ref = tf.cast(truncate(ref), tf.int64) + 1

# print(hyp.eval())
# print(ref.eval())

max_value = tf.reduce_max(tf.concat(0, [hyp, ref]))
tf.assert_greater_equal(max_value ** 4, 2** 63 - 1)

ngrams = [
    (lambda s: s),
    (lambda s: s[:-1] * max_value + s[1:]),
    (lambda s: s[:-2] * max_value ** 2 + s[1:-1] * max_value + s[2:]),
    (lambda s: s[:-3] * max_value ** 3 + s[1:-2] * max_value ** 2 + s[2:-1] * max_value + s[3:]),
]

score = tf.constant(0.0)

for ngram in ngrams:
    hyp_ngrams = ngram(hyp)
    ref_ngrams = ngram(ref)

    hyp_plus_ref = tf.unique_with_counts(tf.concat(0, [hyp_ngrams, ref_ngrams])).y
    # print(hyp_plus_ref.eval())

    hyp_ = tf.concat(0, [hyp_ngrams, hyp_plus_ref])
    ref_ = tf.concat(0, [ref_ngrams, hyp_plus_ref])

    hyp_ = tf.nn.top_k(hyp_, tf.shape(hyp_)[0]).values
    ref_ = tf.nn.top_k(ref_, tf.shape(ref_)[0]).values

    hyp_counts = tf.unique_with_counts(hyp_).count - 1
    ref_counts = tf.unique_with_counts(ref_).count - 1

    # print(hyp_counts.eval())
    # print(ref_counts.eval())

    numerator = tf.cast(tf.reduce_sum(tf.minimum(hyp_counts, ref_counts)), tf.float32) + 1.0
    denominator = tf.cast(tf.reduce_sum(hyp_counts), tf.float32) + 1.0

    score += tf.log(numerator / denominator) / 4

hyp_len = tf.cast(tf.shape(hyp), tf.float32)
ref_len = tf.cast(tf.shape(ref), tf.float32)

bp = tf.minimum(1.0, tf.exp(1.0 - ref_len / hyp_len))

bleu = tf.exp(score) * bp
print(bleu.eval())