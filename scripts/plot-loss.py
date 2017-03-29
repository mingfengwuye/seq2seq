#!/usr/bin/env python3

import argparse
from matplotlib import pyplot as plt
import re

parser = argparse.ArgumentParser()
parser.add_argument('log_files', nargs='+')
parser.add_argument('--output')
parser.add_argument('--max-steps', type=int, default=0)
parser.add_argument('--min-steps', type=int, default=0)
parser.add_argument('--labels', nargs='+')
parser.add_argument('--plot', nargs='+', default=('train', 'dev'))
parser.add_argument('--average', type=int, nargs='+')
parser.add_argument('--smooth', type=int)

args = parser.parse_args()
args.plot = [x.lower() for x in args.plot]

if args.average:
    assert sum(args.average) == len(args.log_files)

n = len(args.average) if args.average else len(args.log_files)

if args.labels:
    if len(args.labels) != n:
        raise Exception('error: wrong number of labels')
    labels = args.labels
else:
    labels = ['model {}'.format(i) for i in range(1, n + 1)]

data = []

for log_file in args.log_files:
    current_step = 0

    dev_perplexities = []
    train_perplexities = []
    bleu_scores = []
    ter_scores = []

    with open(log_file) as f:
        for line in f:
            m = re.search('step (\d+)', line)
            if m:
                current_step = int(m.group(1))

            if 0 < args.max_steps < current_step:
                break
            if current_step < args.min_steps:
                continue

            m = re.search(r'eval: loss (-?\d+.\d+)', line)
            if m and not any(step == current_step for step, _ in dev_perplexities):
                perplexity = float(m.group(1))
                dev_perplexities.append((current_step, perplexity))
                continue

            m = re.search(r'loss (-?\d+.\d+)$', line)
            if m and not any(step == current_step for step, _ in train_perplexities):
                perplexity = float(m.group(1))
                train_perplexities.append((current_step, perplexity))

            m = re.search(r'bleu=(\d+\.\d+)', line)
            m = m or re.search(r'score=(\d+\.\d+)', line)
            if m and not any(step == current_step for step, _ in bleu_scores):
                bleu_score = float(m.group(1))
                bleu_scores.append((current_step, bleu_score))

            m = re.search(r'ter=(\d+\.\d+)', line)
            m = m or re.search(r'score=(\d+\.\d+)', line)
            if m and not any(step == current_step for step, _ in ter_scores):
                ter_score = float(m.group(1))
                ter_scores.append((current_step, ter_score))

    data.append((dev_perplexities, train_perplexities, bleu_scores, ter_scores))


if args.average:
    new_data = []

    i = 0
    for n in args.average:
        data_ = zip(*data[i:i + n])
        i += n

        def avg(data_):
            dicts = [dict(l) for l in data_]
            keys = set.intersection(*[set(d.keys()) for d in dicts])
            data_ = {k: (sum(d[k] for d in dicts) / n) for k in keys}
            data_ = sorted(list(data_.items()))

            if args.smooth is not None and args.smooth > 1:
                k = args.smooth
                data_ = [(data_[i*k][0], sum(x for _, x in data_[i*k:(i+1)*k]) / k) for i in range(len(data_) // k)]

            return data_

        new_data.append(list(map(avg, data_)))
    data = new_data


for label, data_ in zip(labels, data):
    dev_perplexities, train_perplexities, bleu_scores, ter_scores = data_

    if 'bleu' in args.plot and bleu_scores:
        plt.plot(*zip(*bleu_scores), ':', label=' '.join([label, 'BLEU']))
    if 'ter' in args.plot and ter_scores:
        plt.plot(*zip(*ter_scores), ':', label=' '.join([label, 'TER']))
    if 'dev' in args.plot and dev_perplexities:
        plt.plot(*zip(*dev_perplexities), '--', label=' '.join([label, 'dev loss']))
    if 'train' in args.plot and train_perplexities:
        plt.plot(*zip(*train_perplexities), label=' '.join([label, 'train loss']))

legend = plt.legend(loc='best', shadow=True)

if args.output is not None:
    plt.savefig(args.output)
else:
    plt.show()
