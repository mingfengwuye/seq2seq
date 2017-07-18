#!/usr/bin/env python3

import argparse
from translate.utils import reverse_edits

parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('edits')
parser.add_argument('--not-strict', action='store_false', dest='strict')
parser.add_argument('--no-fix', action='store_false', dest='fix')

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.source) as src_file, open(args.edits) as edit_file:
        for source, edits in zip(src_file, edit_file):
            target = reverse_edits(source.strip('\n'), edits.strip('\n'), strict=args.strict, fix=args.fix)
            print(target)
