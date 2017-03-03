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

# scripts/prepare-data.py experiments/post_editing/data/raw/train src pe mt experiments/post_editing/data/ --no-tokenize \
# --dev-corpus experiments/post_editing/data/raw/dev --test-corpus experiments/post_editing/data/raw/test


scripts/extract-edits.py experiments/post_editing/raw_data/APE16/train.{mt,pe} > experiments/post_editing/raw_data/APE16/train.edits
scripts/extract-edits.py experiments/post_editing/raw_data/APE16/dev.{mt,pe} > experiments/post_editing/raw_data/APE16/dev.edits
scripts/extract-edits.py experiments/post_editing/raw_data/APE16/test.{mt,pe} > experiments/post_editing/raw_data/APE16/test.edits
scripts/extract-edits.py experiments/post_editing/raw_data/APE17/train.{mt,pe} > experiments/post_editing/raw_data/APE17/train.edits
scripts/extract-edits.py experiments/post_editing/raw_data/APE17/dev.{mt,pe} > experiments/post_editing/raw_data/APE17/dev.edits


scripts/prepare-data.py experiments/post_editing/raw_data/APE16/train src pe mt edits experiments/post_editing/data_en_de/ \
--no-tokenize \
--dev-corpus experiments/post_editing/raw_data/APE16/dev \
--test-corpus experiments/post_editing/raw_data/APE16/test


scripts/prepare-data.py experiments/post_editing/raw_data/APE17/train src pe mt edits experiments/post_editing/data_de_en/ \
--no-tokenize --dev-size 2000 --dev-prefix train-dev --test-prefix dev \
--test-corpus experiments/post_editing/raw_data/APE16/dev


cat raw_data/artificial/500K.mt > data_en_de_plus/concat.mt
cat raw_data/artificial/500K.pe > data_en_de_plus/concat.pe
cat raw_data/artificial/500K.src > data_en_de_plus/concat.src
cat raw_data/APE16/train.mt >> data_en_de_plus/concat.mt
cat raw_data/APE16/train.pe >> data_en_de_plus/concat.pe
cat raw_data/APE16/train.src >> data_en_de_plus/concat.src