#!/usr/bin/env python3

import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('log_file')
parser.add_argument('--dev-prefix', default='dev')
parser.add_argument('--score', default='ter', choices=('ter', 'wer', 'bleu'))

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.log_file) as log_file:
        lines = log_file.readlines()

        scores = [re.findall('(\w+)=(\d+.\d+)', line) for line in lines]
        scores = [{k: float(v) for k, v in score} for score in scores if score]

        def key(d):
            score = d.get(args.score.lower()) or d.get('score')
            if args.score in ('wer', 'ter'):
                score = -score
            return score

        best = max(scores, key=key)

        key = args.score if args.score in best else 'score'

        main_score = '{}={} '.format(key, best[key])
        print(main_score + ' '.join('{}={}'.format(k, v)for k, v in sorted(best.items()) if k != key))
