#!/usr/bin/env bash

# Summary of the task at: http://www.statmt.org/wmt16/ape-task.html
# interesting papers on the subject:
# https://arxiv.org/abs/1606.07481:   multi-source, no additional data, predicts a sequence of edits, good results
# https://arxiv.org/abs/1605.04800:   merges two mono-source models, lots of additional (parallel data), creates
# synthetic PE data by using back-translation

# Ideas: multi-source, additional monolingual data, and some parallel data (not too much)
# pre-train with auto-encoder / word-embeddings / multi-task
# + Beam-search + LM + Ensemble
# other info ? e.g. POS tags
# Finetune with REINFORCE

# xz -dkf commoncrawl.de.xz --verbose

scripts/extract-edits.py experiments/APE16/raw_data/train.{mt,pe} > experiments/APE16/raw_data/train.edits
scripts/extract-edits.py experiments/APE16/raw_data/dev.{mt,pe} > experiments/APE16/raw_data/dev.edits
scripts/extract-edits.py experiments/APE16/raw_data/test.{mt,pe} > experiments/APE16/raw_data/test.edits
scripts/extract-edits.py experiments/APE16/raw_data/500K.{mt,pe} > experiments/APE16/raw_data/500K.edits

mkdir -p experiments/APE16/data experiments/APE16/data_plus

cp experiments/APE16/raw_data/500K.mt experiments/APE16/data_plus/concat.mt
cp experiments/APE16/raw_data/500K.pe experiments/APE16/data_plus/concat.pe
cp experiments/APE16/raw_data/500K.src experiments/APE16/data_plus/concat.src
cp experiments/APE16/raw_data/500K.edits experiments/APE16/data_plus/concat.edits
cat experiments/APE16/raw_data/train.mt >> experiments/APE16/data_plus/concat.mt
cat experiments/APE16/raw_data/train.pe >> experiments/APE16/data_plus/concat.pe
cat experiments/APE16/raw_data/train.src >> experiments/APE16/data_plus/concat.src
cat experiments/APE16/raw_data/train.edits >> experiments/APE16/data_plus/concat.edits

scripts/prepare-data.py experiments/APE16/raw_data/train src pe mt edits experiments/APE16/data --no-tokenize \
--dev-corpus experiments/APE16/raw_data/dev \
--test-corpus experiments/APE16/raw_data/test

scripts/prepare-data.py experiments/APE16/data_plus/concat src pe mt edits experiments/APE16/data_plus --no-tokenize \
--dev-corpus experiments/APE16/raw_data/dev \
--test-corpus experiments/APE16/raw_data/test --shuffle