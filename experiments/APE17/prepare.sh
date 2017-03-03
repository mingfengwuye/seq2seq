#!/usr/bin/env bash

mkdir experiments/APE17/data

scripts/extract-edits.py experiments/APE17/raw_data/train.{mt,pe} > experiments/APE17/raw_data/train.edits
scripts/extract-edits.py experiments/APE17/raw_data/dev.{mt,pe} > experiments/APE17/raw_data/dev.edits

scripts/prepare-data.py experiments/APE17/raw_data/train src pe mt edits experiments/APE17/data \
--no-tokenize --dev-size 2000 --dev-prefix train-dev --test-prefix dev \
--test-corpus experiments/APE17/raw_data/dev