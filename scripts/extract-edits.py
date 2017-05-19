#!/usr/bin/env python3

import argparse
import functools
import sys

parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('target')
parser.add_argument('--subs', action='store_true')
parser.add_argument('--ops-only', action='store_true')
parser.add_argument('--words-only', action='store_true')
parser.add_argument('--char-level', action='store_true')
parser.add_argument('--sub-cost', type=float, default=1.0)
parser.add_argument('--del-cost', type=float, default=1.0)
parser.add_argument('--ins-cost', type=float, default=1.0)
parser.add_argument('--cache-size', type=int, default=512)


if __name__ == '__main__':
    args = parser.parse_args()
    assert not args.words_only or not args.ops_only

    @functools.lru_cache(maxsize=args.cache_size)
    def levenshtein(src, trg, sub_cost=1.0, del_cost=1.0, ins_cost=1.0):
        params = {'sub_cost': sub_cost, 'del_cost': del_cost, 'ins_cost': ins_cost}

        if len(src) == 0:
            return len(trg), [('insert', w) for w in trg]
        elif len(trg) == 0:
            return len(src), ['delete' for _ in src]

        insert = levenshtein(src, trg[1:], **params)
        delete = levenshtein(src[1:], trg, **params)

        res = [
            (ins_cost + insert[0], [('insert', trg[0])] + insert[1]),
            (del_cost + delete[0], ['delete'] + delete[1])
        ]

        if src[0] == trg[0]:
            keep = levenshtein(src[1:], trg[1:], **params)
            res.append((keep[0], [('keep', trg[0])] + keep[1]))
        else:
            keep = levenshtein(src[1:], trg[1:], **params)
            res.append((sub_cost + keep[0], [('sub', trg[0])] + keep[1]))

        return min(res, key=lambda p: p[0])

    sys.setrecursionlimit(10000)

    with open(args.source) as src_file, open(args.target) as trg_file:
        for src_line, trg_line in zip(src_file, trg_file):
            if args.char_level:
                src_words = tuple(x if x.strip() else '<SPACE>' for x in src_line.strip('\n'))
                trg_words = tuple(x if x.strip() else '<SPACE>' for x in trg_line.strip('\n'))
            else:
                src_words = tuple(src_line.split())
                trg_words = tuple(trg_line.split())

            if not args.subs:
                args.sub_cost = float('inf')

            try:
                _, ops = levenshtein(src_words, trg_words, sub_cost=args.sub_cost,
                                     del_cost=args.del_cost, ins_cost=args.ins_cost)
            except KeyboardInterrupt:
                sys.exit()

            edits = []
            for op in ops:
                if op == 'delete':
                    edit = '<DEL>'
                elif op[0] == 'keep':
                    if args.words_only:
                        edit = op[1]
                    else:
                        edit = '<KEEP>'
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
