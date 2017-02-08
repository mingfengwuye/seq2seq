#!/usr/bin/env python3
import argparse
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('filename')

args = parser.parse_args()


word_counts = Counter()
char_counts = Counter()

word_dict = Counter()
char_dict = Counter()

line_dict = Counter()

with open(args.filename) as f:
    for line in f:
        line = line.strip()

        words = line.split()
        chars = line.replace(' ', '')

        word_counts[len(words)] += 1
        char_counts[len(chars)] += 1

        word_dict += Counter(words)
        char_dict += Counter(chars)

        line_dict[line] += 1


def avg(counter):
    return sum(counter.values()) / len(counter)

def total(counter):
    return sum(counter.values())

def stats(counter):
    total = sum(counter.values())
    unique = len(counter)
    min_ = min(counter.values())
    max_ = max(counter.values())

    l = [[k] * v for k, v in sorted(counter.items())]
    l = [x for l_ in l for x in l_]

    mode = l[len(l) // 2]
    avg = total / unique

    tenth = l[len(l) // 10]
    nineth = l[len(l) * 9 // 10]

    return 'total={} unique={} min={}, max={}, avg={}, mode={}, 10%={}, 90%={}'.format(
        total, unique, min_, max_, avg, mode, tenth, nineth)


print('Words per line:', stats(word_counts))
print('Chars per line:', stats(char_counts))

print('Lines:', stats(line_dict))
print('Words:', stats(word_dict))
print('Chars:', stats(char_dict))


# print('total lines:', total(lines))
# print('total unique lines:', len(lines))
# print('avg line count:', avg(lines))
# print('max line count:', max(lines.values()))
#
# print('total words:', sum(word_counts.values()))
# print('total chars:', sum(char_counts.values()))
#
#
# print('avg words per line:', sum(word_counts.values()) / len(word_counts))
# print('avg chars per line:', sum(char_counts.values()) / len(char_counts))
#
# print('max words per line:', max(word_counts.values()))
# print('max chars per line:', max(char_counts.values()))
#
#
# print('avg word count')