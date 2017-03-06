#!/usr/bin/env python3

import argparse
from translate.evaluation import corpus_bleu, corpus_ter, corpus_tercom, corpus_wer
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('target')
parser.add_argument('--bleu', action='store_true')
parser.add_argument('--pyter', action='store_true')
parser.add_argument('--ter', action='store_true')
parser.add_argument('--wer', action='store_true')
parser.add_argument('--all', '-a', action='store_true')

parser.add_argument('--case-insensitive', '-i', action='store_true')


if __name__ == '__main__':
    args = parser.parse_args()

    if not any([args.all, args.wer, args.ter, args.bleu, args.pyter]):
        args.all = True

    if args.all:
        args.wer = args.tercom = args.bleu = True

    with open(args.source) as src_file, open(args.target) as trg_file:
        if args.case_insensitive:
            hypotheses = [line.strip().lower() for line in src_file]
            references = [line.strip().lower() for line in trg_file]
        else:
            hypotheses = [line.strip() for line in src_file]
            references = [line.strip() for line in trg_file]

        scores = OrderedDict()
        if args.bleu:
            scores['bleu'], _ = corpus_bleu(hypotheses, references)
        if args.wer:
            scores['wer'], _ = corpus_wer(hypotheses, references)
        if args.ter:
            scores['ter'], _ = corpus_tercom(hypotheses, references)
        if args.pyter:
            scores['pyter'], _ = corpus_ter(hypotheses, references)

        print(' '.join('{}={:.2f}'.format(k, v) for k, v in scores.items()))
