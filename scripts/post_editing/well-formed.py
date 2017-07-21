#!/usr/bin/env python3
import sys
from signal import signal, SIGPIPE, SIG_DFL

signal(SIGPIPE, SIG_DFL)

punk = '.!?'

def is_well_formed(line):
    if len(line) < 31:
        return False

    x = line[0]
    if not x.isalpha() or not x.isupper():
        return False
    if not line[-2] in punk:  # last character is '\n'
        return False

    i = 0
    for c in line:
        if c.isalpha():
            i += 1
        if i == 30:
            return True
    return False


if __name__ == '__main__':
    for line in sys.stdin:
        if is_well_formed(line):
            sys.stdout.write(line)
