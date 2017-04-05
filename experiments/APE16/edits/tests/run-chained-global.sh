#!/usr/bin/env bash

./seq2seq.sh current/tests/chained-global.yaml --train -v --purge
./seq2seq.sh current/tests/chained-syn-global.yaml --train -v --purge