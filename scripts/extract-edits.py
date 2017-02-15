#!/usr/bin/env python3

import argparse
import functools

parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('target')


@functools.lru_cache(maxsize=1024)
def levenshtein(src, trg):
    if len(src) == 0:
        return len(trg), [('insert', w) for w in trg]
    elif len(trg) == 0:
        return len(src), ['delete' for _ in src]

    insert = levenshtein(src, trg[1:])
    delete = levenshtein(src[1:], trg)

    res = [
        (1 + insert[0], [('insert', trg[0])] + insert[1]),
        (1 + delete[0], ['delete'] + delete[1])
    ]

    if src[0] == trg[0]:
        keep = levenshtein(src[1:], trg[1:])
        res.append((keep[0], ['keep'] + keep[1]))

    return min(res, key=lambda p: p[0])


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.source) as src_file, open(args.target) as trg_file:
        for src_line, trg_line in zip(src_file, trg_file):
            src_words = tuple(src_line.split())
            trg_words = tuple(trg_line.split())

            _, edits = levenshtein(src_words, trg_words)

            edits = [
                '<KEEP>' if op == 'keep' else
                '<DEL>' if op == 'delete' else
                op[1] for op in edits]

            print(' '.join(edits))
