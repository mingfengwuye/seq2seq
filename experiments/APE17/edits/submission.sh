#!/usr/bin/env bash

mkdir -p WMT17/de-en
./seq2seq.sh experiments/APE17/edits/forced/config.yaml -v --decode experiments/APE17/raw_data/test.2017.mt |\
python3 -c "import sys; print('\n'.join('forced\t{}\t{}'.format(i, line.strip()) for i, line in enumerate(sys.stdin, 1)))" \
> WMT17/de-en/LIG_forced_CONTRASTIVE

./seq2seq.sh experiments/APE17/edits/chained/config.yaml -v --decode experiments/APE17/raw_data/test.2017.{mt,src} |\
python3 -c "import sys; print('\n'.join('chained\t{}\t{}'.format(i, line.strip()) for i, line in enumerate(sys.stdin, 1)))" \
> WMT17/de-en/LIG_chained_CONTRASTIVE

./seq2seq.sh experiments/APE17/edits/chained_syn/config.yaml -v --decode experiments/APE17/raw_data/test.2017.{mt,src} |\
python3 -c "import sys; print('\n'.join('chained_syn\t{}\t{}'.format(i, line.strip()) for i, line in enumerate(sys.stdin, 1)))" \
> WMT17/de-en/LIG_chained_syn_PRIMARY
