#!/usr/bin/env bash

scripts/detruecase.perl < $1 > /tmp/source.re
scripts/score.py /tmp/source.re $2
rm -f /tmp/source.re
