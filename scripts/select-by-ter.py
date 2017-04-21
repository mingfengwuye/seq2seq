#!/usr/bin/env python3

import argparse
import numpy as np
import random
import sys

parser = argparse.ArgumentParser()

parser.add_argument('ref_vectors')
parser.add_argument('vectors')
parser.add_argument('-n', type=int, default=500000)
parser.add_argument('-k', type=int, default=1)
parser.add_argument('-m', type=int, default=1000)

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.ref_vectors) as f:
        ref_vectors = [np.array([float(x) for x in line.split(',')]) for line in f]
    with open(args.vectors) as f:
        vectors = [np.array([float(x) for x in line.split(',')]) for line in f]
        vectors = list(enumerate(vectors))

    n = 0

    while n < args.n and len(vectors) > 0:
        vector = ref_vectors[n % len(ref_vectors)]
        n += 1

        def key(i):
            return np.sum((vector - vectors[i][1]) ** 2)

        indices = random.sample(range(len(vectors)), k=args.m)

        if args.k > 1:
            indices = sorted(indices, key=key)[:args.k]
        else:
            indices = [min(indices, key=key)]

        for i in indices:
            sys.stdout.write(str(vectors[i][0]) + '\n')
        sys.stdout.flush()

        for i in sorted(indices, reverse=True):
            del vectors[i]
