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
        scores = {}
        current_step = 0

        for line in log_file:
            m = re.search('step (\d+)', line)
            if m:
                current_step = int(m.group(1))
                continue

            m = re.findall('(\w+)=(\d+.\d+)', line)
            if m:
                scores_ = {k: float(v) for k, v in m}
                scores.setdefault(current_step, scores_)

        def key(d):
            score = d.get(args.score.lower()) or d.get('score')
            if args.score in ('wer', 'ter'):
                score = -score
            return score

        step, best = max(scores.items(), key=lambda p: key(p[1]))

        key = args.score if args.score in best else 'score'

        main_score = '{}={:.2f} step={} '.format(key, best[key], step)
        print(main_score + ' '.join('{}={:.2f}'.format(k, v) for k, v in sorted(best.items()) if k != key))
