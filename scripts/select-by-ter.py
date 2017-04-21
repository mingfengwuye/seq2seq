#!/usr/bin/env python3

import argparse
import numpy as np
import random

parser = argparse.ArgumentParser()

parser.add_argument('ref_vectors')
parser.add_argument('vectors')
parser.add_argument('-n', type=int, default=500000)
parser.add_argument('-k', type=int, default=10)

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.ref_vectors) as f:
        ref_vectors = [np.array([float(x) for x in line.split(',')]) for line in f]
    with open(args.vectors) as f:
        vectors = [np.array([float(x) for x in line.split(',')]) for line in f]

    n = 0

    while n < args.n and len(vectors) > 0:
        vector = ref_vectors[n % len(ref_vectors)]
        n += 1

        v = random.sample(range(len(vectors)), k=10000)
        v.sort(key=lambda i: np.sum((vector - vectors[i]) ** 2))

        indices = v[:args.k]

        print('\n'.join(map(str, indices)))

        for i in sorted(indices, reverse=True):
            del vectors[i]
