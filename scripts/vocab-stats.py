#!/usr/bin/env python3

import sys
import argparse
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('--reference')
parser.add_argument('--max', type=int)

args = parser.parse_args()

if args.reference is not None:
    with open(args.reference) as ref_file:
        ref_words = [Counter(line.split()) for line in ref_file]
else:
    ref_words = None

total = Counter()
ok = Counter()

with open(args.source) as src_file:
    for i, line in enumerate(src_file):
        if ref_words and i >= len(ref_words):
            break

        words = Counter(line.split())
        total += words

        if ref_words:
            ref = ref_words[i]
            ok += Counter(dict((w, min(c, ref[w])) for w, c in words.items()))

total_count = sum(total.values())

precision_header = ' {:8}'.format('precision') if args.reference else ''
header = '{:15} {:8} {:8}'.format('word', 'count', 'percentage') + precision_header
print(header)

for w, c in total.most_common(args.max):
    precision = ' {:8.2f}%'.format(100 * ok[w] / c) if args.reference else ''

    print('{:15} {:8} {:8.2f}%'.format(w, c, 100 * c / total_count) + precision)
