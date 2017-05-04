#!/usr/bin/env bash

raw_data=experiments/APE17/raw_data

./scripts/extract-ter-vectors.py --precision 5 ${raw_data}/mono.{mt,en} > ${raw_data}/mono.mt.ter
./scripts/extract-ter-vectors.py --precision 5 ${raw_data}/train.{mt,pe} > ${raw_data}/train.mt.ter

./scripts/select-by-ter.py ${raw_data}/{train,mono}.mt.ter -n 500000 -k 1 -m 1000 > ${raw_data}/mono.500k.index
./scripts/select-by-ter.py ${raw_data}/{train,mono}.mt.ter -n 4000000 -k 1 -m 1000 > ${raw_data}/mono.4M.index

./scripts/select-by-index.py ${raw_data}/mono.500k.index < ${raw_data}/mono.mt > ${raw_data}/mono.500k.mt
./scripts/select-by-index.py ${raw_data}/mono.500k.index < ${raw_data}/mono.en > ${raw_data}/mono.500k.pe
./scripts/select-by-index.py ${raw_data}/mono.500k.index < ${raw_data}/mono.src > ${raw_data}/mono.500k.src

./scripts/select-by-index.py ${raw_data}/mono.4M.index < ${raw_data}/mono.mt > ${raw_data}/mono.4M.mt
./scripts/select-by-index.py ${raw_data}/mono.4M.index < ${raw_data}/mono.en > ${raw_data}/mono.4M.pe
./scripts/select-by-index.py ${raw_data}/mono.4M.index < ${raw_data}/mono.src > ${raw_data}/mono.4M.src

./scripts/noisify.py ${raw_data}/train.{mt,pe} < ${raw_data}/mono.en > ${raw_data}/mono.noisy.en
