import functools
import subprocess
import tempfile
import math
import numpy as np
import re
import os

from collections import Counter, OrderedDict


def sentence_bleu(hypothesis, reference, smoothing=True, order=4, **kwargs):
    """
    Compute sentence-level BLEU score between a translation hypothesis and a reference.

    :param hypothesis: list of tokens or token ids
    :param reference: list of tokens or token ids
    :param smoothing: apply smoothing (recommended, especially for short sequences)
    :param order: count n-grams up to this value of n.
    :param kwargs: additional (unused) parameters
    :return: BLEU score (float)
    """
    log_score = 0

    if len(hypothesis) == 0:
        return 0

    for i in range(order):
        hyp_ngrams = Counter(zip(*[hypothesis[j:] for j in range(i + 1)]))
        ref_ngrams = Counter(zip(*[reference[j:] for j in range(i + 1)]))

        numerator = sum(min(count, ref_ngrams[bigram]) for bigram, count in hyp_ngrams.items())
        denominator = sum(hyp_ngrams.values())

        if smoothing:
            numerator += 1
            denominator += 1

        score = numerator / denominator

        if score == 0:
            log_score += float('-inf')
        else:
            log_score += math.log(score) / order

    bp = min(1, math.exp(1 - len(reference) / len(hypothesis)))

    return math.exp(log_score) * bp


def score_function_decorator(reversed=False):
    def decorator(func):
        func.reversed = reversed
        return func
    return decorator


def corpus_bleu(hypotheses, references, smoothing=False, order=4, **kwargs):
    """
    Computes the BLEU score at the corpus-level between a list of translation hypotheses and references.
    With the default settings, this computes the exact same score as `multi-bleu.perl`.

    All corpus-based evaluation functions should follow this interface.

    :param hypotheses: list of strings
    :param references: list of strings
    :param smoothing: apply +1 smoothing
    :param order: count n-grams up to this value of n. `multi-bleu.perl` uses a value of 4.
    :param kwargs: additional (unused) parameters
    :return: score (float), and summary containing additional information (str)
    """
    total = np.zeros((4,))
    correct = np.zeros((4,))

    hyp_length = 0
    ref_length = 0

    for hyp, ref in zip(hypotheses, references):
        hyp = hyp.split()
        ref = ref.split()

        hyp_length += len(hyp)
        ref_length += len(ref)

        for i in range(order):
            hyp_ngrams = Counter(zip(*[hyp[j:] for j in range(i + 1)]))
            ref_ngrams = Counter(zip(*[ref[j:] for j in range(i + 1)]))

            total[i] += sum(hyp_ngrams.values())
            correct[i] += sum(min(count, ref_ngrams[bigram]) for bigram, count in hyp_ngrams.items())

    if smoothing:
        total += 1
        correct += 1

    scores = correct / total

    score = math.exp(
        sum(math.log(score) if score > 0 else float('-inf') for score in scores) / order
    )

    bp = min(1, math.exp(1 - ref_length / hyp_length))
    bleu = 100 * bp * score

    return bleu, 'penalty={:.3f} ratio={:.3f}'.format(bp, hyp_length / ref_length)


@score_function_decorator(reversed=True)
def corpus_ter(hypotheses, references, case_sensitive=True, **kwargs):
    with tempfile.NamedTemporaryFile('w') as hypothesis_file, tempfile.NamedTemporaryFile('w') as reference_file:
        for i, (hypothesis, reference) in enumerate(zip(hypotheses, references)):
            hypothesis_file.write('{} ({})\n'.format(hypothesis, i))
            reference_file.write('{} ({})\n'.format(reference, i))
        hypothesis_file.flush()
        reference_file.flush()

        cmd = ['java', '-jar', 'scripts/tercom.jar', '-h', hypothesis_file.name, '-r', reference_file.name]
        if case_sensitive:
            cmd.append('-s')

        output = subprocess.check_output(cmd).decode()

        error = re.findall(r'Total TER: (.*?) ', output, re.MULTILINE)[0]
        return float(error) * 100, ''


@score_function_decorator(reversed=True)
def corpus_wer(hypotheses, references, **kwargs):
    scores = [
        levenhstein(tuple(hyp.split()), tuple(ref.split())) / len(ref.split())
        for hyp, ref in zip(hypotheses, references)
    ]

    score = 100 * sum(scores) / len(scores)

    hyp_length = sum(len(hyp.split()) for hyp in hypotheses)
    ref_length = sum(len(ref.split()) for ref in references)

    return score, 'ratio={:.3f}'.format(hyp_length / ref_length)


def corpus_scores(hypotheses, references, main='bleu', **kwargs):
    bleu_score, summary = corpus_bleu(hypotheses, references)
    # ter, _ = corpus_ter(hypotheses, references)
    ter, _ = corpus_ter(hypotheses, references)
    wer, _ = corpus_wer(hypotheses, references)

    scores = OrderedDict([('bleu', bleu_score), ('ter', ter), ('wer', wer)])
    main_score = scores[main]
    summary = ' '.join([summary] + ['{}={:.2f}'.format(k, v)
                                    for k, v in scores.items() if k != main])

    return main_score, summary


@score_function_decorator(reversed=True)
def corpus_scores_ter(*args, **kwargs):
    return corpus_scores(*args, main='ter', **kwargs)


@score_function_decorator(reversed=True)
def corpus_scores_wer(*args, **kwargs):
    return corpus_scores(*args, main='wer', **kwargs)


corpus_scores_bleu = corpus_scores


@functools.lru_cache(maxsize=1024)
def levenhstein(src, trg):
    # Dynamic programming by memoization
    if len(src) == 0:
        return len(trg)
    elif len(trg) == 0:
        return len(src)

    return min(
        int(src[0] != trg[0]) + levenhstein(src[1:], trg[1:]),
        1 + levenhstein(src[1:], trg),
        1 + levenhstein(src, trg[1:])
    )


def tercom_statistics(hypotheses, references, case_sensitive=True, **kwargs):
    with tempfile.NamedTemporaryFile('w') as hypothesis_file, tempfile.NamedTemporaryFile('w') as reference_file:
        for i, (hypothesis, reference) in enumerate(zip(hypotheses, references)):
            hypothesis_file.write('{} ({})\n'.format(hypothesis, i))
            reference_file.write('{} ({})\n'.format(reference, i))
        hypothesis_file.flush()
        reference_file.flush()

        filename = tempfile.mktemp()

        cmd = ['java', '-jar', 'scripts/tercom.jar', '-h', hypothesis_file.name, '-r', reference_file.name,
               '-o', 'sum', '-n', filename]
        if case_sensitive:
            cmd.append('-s')

        output = open('/dev/null', 'w')
        subprocess.call(cmd, stdout=output, stderr=output)

    with open(filename + '.sum') as f:
        fields = ['DEL', 'INS', 'SUB', 'SHIFT', 'WORD_SHIFT', 'ERRORS', 'REF_WORDS']

        stats = []
        for line in f:
            values = line.strip().split('|')
            if len(values) != 9:
                continue
            try:
                # values = np.array([float(x) for x in values[1:]])
                values = dict(zip(fields, map(float, values[1:])))
            except ValueError:
                continue

            stats.append(values)

        assert len(stats) == len(hypotheses) + 1

        total = stats[-1]
        stats = stats[:-1]
        total = {k: v / len(stats) for k, v in total.items()}

    os.remove(filename + '.sum')
    return total, stats
