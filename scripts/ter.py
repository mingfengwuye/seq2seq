#!/usr/bin/env python3

import argparse
from translate import pyter

parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('target')

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.source) as src_file, open(args.target) as trg_file:
        ter = 0
        lines = 0

        for src_line, trg_line in zip(src_file, trg_file):
            ter += pyter.ter(src_line.split(), trg_line.split())
            lines += 1

        ter /= lines

        print('TER: {:.2f}'.format(100 * ter))