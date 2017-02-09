#!/usr/bin/env python3
import argparse
from collections import Counter, namedtuple, OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('--lower', action='store_true')
parser.add_argument('--count-whitespaces', action='store_true')
parser.add_argument('-c', '--chars', action='store_true')
parser.add_argument('-l', '--lines', action='store_true')
parser.add_argument('-w', '--words', action='store_true')

args = parser.parse_args()

if not args.chars and not args.lines and not args.words:
    args.chars = args.lines = args.words = True

word_counts = Counter()
char_counts = Counter()

word_dict = Counter()
char_dict = Counter()

line_dict = Counter()

with open(args.filename) as f:
    for line in f:
        if args.lower:
            line = line.lower()

        if args.words:
            words = line.split()
            word_counts[len(words)] += 1
            for word in words:
                word_dict[word] += 1

        if args.chars:
            chars = line
            if not args.count_whitespaces:
                chars = line.strip().replace(' ', '')

            char_counts[len(line)] += 1
            for char in line:
                char_dict[char] += 1

        if args.lines:
            line_dict[line] += 1


def info_dict(title, counter):
    total = sum(counter.values())
    unique = len(counter)
    avg = total / unique
    min_ = min(counter.values())
    max_ = max(counter.values())

    cumulative_count = 0
    coverage = OrderedDict([(90, 0), (95, 0), (99, 0)])

    for i, pair in enumerate(counter.most_common(), 1):
        _, count = pair
        cumulative_count += count

        for percent, count in coverage.items():
            if count == 0 and cumulative_count >= percent * total / 100:
                coverage[percent] = i

    summary = [
        '{}\n{}'.format(title, '-' * len(title)),
        'Total:   {}'.format(total),
        'Unique:  {}'.format(unique),
        'Minimum: {}'.format(min_),
        'Maximum: {}'.format(max_),
        'Average: {:.1f}'.format(avg)
    ]

    for percent, count in coverage.items():
        summary.append('{}% cov: {}'.format(percent, count))

    return '\n  '.join(summary) + '\n'


def info_lengths(title, counter):
    total = sum(counter.values())
    avg = sum(k * v for k, v in counter.items()) / total

    l = [[k] * v for k, v in sorted(counter.items())]
    l = [x for l_ in l for x in l_]
    mode = l[len(l) // 2]

    tenth = l[len(l) // 10]
    nineth = l[len(l) * 9 // 10]

    summary = [
        '{}\n{}'.format(title, '-' * len(title)),
        'Minimum: {}'.format(min(counter)),
        'Maximum: {}'.format(max(counter)),
        'Average: {:.1f}'.format(avg),
        'Mode:    {}'.format(mode),
        '<10%:    {}'.format(tenth),
        '<90%:    {}'.format(nineth),
    ]

    return '\n  '.join(summary) + '\n'

if args.lines:
    total = sum(line_dict.values())
    unique = len(line_dict)
    avg = total / unique
    title = 'Lines'

    summary = [
        '{}\n{}'.format(title, '-' * len(title)),
        'Total:   {}'.format(total),
        'Unique:  {}'.format(unique),
        'Average: {:.2f}'.format(avg)
    ]

    print('\n  '.join(summary) + '\n')

if args.words:
    print(info_lengths('Words per line', word_counts))
    print(info_dict('Words', word_dict))

if args.chars:
    print(info_lengths('Chars per line', char_counts))
    print(info_dict('Chars', char_dict))
