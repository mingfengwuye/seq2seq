#!/usr/bin/env bash

mkdir -p WMT17/en-de
./seq2seq.sh experiments/APE16/edits/mono/forced/config.yaml -v --decode experiments/APE16/raw_data/test.2017.mt |\
python3 -c "import sys; print('\n'.join('forced\t{}\t{}'.format(i, line.strip()) for i, line in enumerate(sys.stdin, 1)))" \
> WMT17/en-de/LIG_forced_CONTRASTIVE

./seq2seq.sh experiments/APE16/edits/chained/forced/config.yaml -v --decode experiments/APE16/raw_data/test.2017.{mt,src} |\
python3 -c "import sys; print('\n'.join('chained\t{}\t{}'.format(i, line.strip()) for i, line in enumerate(sys.stdin, 1)))" \
> WMT17/en-de/LIG_chained_CONTRASTIVE

./seq2seq.sh experiments/APE16/edits/chained/forced_syn/config.yaml -v --decode experiments/APE16/raw_data/test.2017.{mt,src} |\
python3 -c "import sys; print('\n'.join('chained_syn\t{}\t{}'.format(i, line.strip()) for i, line in enumerate(sys.stdin, 1)))" \
> WMT17/en-de/LIG_chained_syn_PRIMARY
