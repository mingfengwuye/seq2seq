#!/usr/bin/env python3

import argparse
import functools

parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('target')
parser.add_argument('--subs', action='store_true')
parser.add_argument('--ops-only', action='store_true')
parser.add_argument('--words-only', action='store_true')

@functools.lru_cache(maxsize=1024)
def levenshtein(src, trg, subs=False):
    if len(src) == 0:
        return len(trg), [('insert', w) for w in trg]
    elif len(trg) == 0:
        return len(src), ['delete' for _ in src]

    insert = levenshtein(src, trg[1:], subs=subs)
    delete = levenshtein(src[1:], trg, subs=subs)

    res = [
        (1 + insert[0], [('insert', trg[0])] + insert[1]),
        (1 + delete[0], ['delete'] + delete[1])
    ]

    if src[0] == trg[0]:
        keep = levenshtein(src[1:], trg[1:], subs=subs)
        res.append((keep[0], ['keep'] + keep[1]))
    elif subs:
        keep = levenshtein(src[1:], trg[1:], subs=subs)
        res.append((1 + keep[0], [('sub', trg[0])] + keep[1]))

    return min(res, key=lambda p: p[0])


if __name__ == '__main__':
    args = parser.parse_args()
    assert not args.words_only or not args.ops_only

    with open(args.source) as src_file, open(args.target) as trg_file:
        for src_line, trg_line in zip(src_file, trg_file):
            src_words = tuple(src_line.split())
            trg_words = tuple(trg_line.split())

            _, ops = levenshtein(src_words, trg_words, subs=args.subs)

            edits = []
            for op in ops:
                if op == 'keep':
                    if args.words_only:
                        edit = '<NONE>'
                    else:
                        edit = '<KEEP>'
                elif op == 'delete':
                    if args.words_only:
                        edit = '<NONE>'
                    else:
                        edit = '<DEL>'
                elif op[0] == 'insert':
                    if args.words_only:
                        edit = op[1]
                    elif args.ops_only:
                        edit = '<INS>'
                    elif args.subs:
                        edit = '<INS>_{}'.format(op[1])
                    else:
                        edit = op[1]
                else:
                    if args.words_only:
                        edit = op[1]
                    elif args.ops_only:
                        edit = '<SUB>'
                    else:
                        edit = '<SUB>_{}'.format(op[1])

                edits.append(edit)

            print(' '.join(edits))
