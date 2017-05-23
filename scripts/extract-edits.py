#!/usr/bin/env python3

import argparse
import sys
import numpy as np

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


def levenshtein(src, trg, sub_cost=1.0, del_cost=1.0, ins_cost=1.0):
    KEEP, SUB, INS, DEL = range(4)
    op_names = 'keep', 'sub', 'insert', 'delete'

    costs = np.zeros((len(trg) + 1, len(src) + 1))
    ops = np.zeros((len(trg) + 1, len(src) + 1), dtype=np.int32)

    costs[0] = range(len(src) + 1)
    costs[:,0] = range(len(trg) + 1)
    ops[0] = DEL
    ops[:,0] = INS

    for i in range(1, len(trg) + 1):
        for j in range(1, len(src) + 1):
            c, op = (sub_cost, SUB) if trg[i - 1] != src[j - 1] else (0, KEEP)

            costs[i,j], ops[i,j] = min([
                (costs[i - 1, j] + ins_cost, INS),
                (costs[i, j - 1] + del_cost, DEL),
                (costs[i - 1, j - 1] + c, op),
            ], key=lambda p: p[0])

    # backtracking
    i, j = len(trg), len(src)
    cost = costs[i, j]

    res = []

    while i > 0 or j > 0:
        op = ops[i, j]
        op_name = op_names[op]

        if op == DEL:
            res.append(op_name)
            j -= 1
        else:
            res.append((op_name, trg[i - 1]))
            i -= 1
            if op != INS:
                j -= 1

    return cost, res[::-1]


if __name__ == '__main__':
    args = parser.parse_args()
    assert not args.words_only or not args.ops_only

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
