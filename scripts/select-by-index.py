#!/usr/bin/env python3

import argparse
import numpy as np
import random
import sys

parser = argparse.ArgumentParser()

parser.add_argument('indices')

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.indices) as f:
        indices = sorted(list(set([int(line) for line in f])))

    for i, line in enumerate(sys.stdin):
        if len(indices) == 0:
            break

        if i == indices[0]:
            indices.pop(0)
            print(line.rstrip('\r\n'))
