#!/usr/bin/env python3

import argparse
import random
import sys

parser = argparse.ArgumentParser()

parser.add_argument('ref_sentences')
parser.add_argument('sentences')
parser.add_argument('-n', type=int, default=500000)
parser.add_argument('-k', type=int, default=1)
parser.add_argument('-m', type=int, default=1000)

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.ref_sentences) as f:
        ref_lengths = [len(line.split()) for line in f]
    with open(args.sentences) as f:
        lengths = [len(line.split()) for line in f]
        lengths = list(enumerate(lengths))

    n = 0

    indices = []

    while n < args.n and len(lengths) > 0:
        length = ref_lengths[n % len(ref_lengths)]
        n += 1

        def key(i):
            return abs(length - lengths[i][1])

        indices_ = random.sample(range(len(lengths)), k=args.m)

        if args.k > 1:
            indices_ = sorted(indices_, key=key)[:args.k]
        else:
            indices_ = [min(indices_, key=key)]

        indices += indices_

        for i in sorted(indices_, reverse=True):
            del lengths[i]

    indices = sorted(list(set(indices)), reverse=True)

    with open(args.sentences) as f:
        for i, line in enumerate(f):
            if len(indices) == 0:
                break

            if i == indices[-1]:
                sys.stdout.write(line)
                sys.stdout.flush()
                indices.pop()
