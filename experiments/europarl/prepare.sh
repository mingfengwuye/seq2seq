#!/usr/bin/env bash

raw_data=data/raw
data_dir=experiments/europarl/data

rm -rf ${data_dir}
mkdir -p ${data_dir}

cat ${raw_data}/{europarl,news-commentary}.fr-en.fr > ${data_dir}/concat.fr
cat ${raw_data}/{europarl,news-commentary}.fr-en.en > ${data_dir}/concat.en

scripts/prepare-data.py ${data_dir}/concat fr en ${data_dir} --vocab-size 30000 --shuffle --seed 1234 \
--unescape-special-chars --normalize-punk

scripts/unescape-special-chars.perl < ${raw_data}/ntst1213.fr-en.fr > ${data_dir}/dev.fr
scripts/unescape-special-chars.perl < ${raw_data}/ntst1213.fr-en.en > ${data_dir}/dev.en
scripts/unescape-special-chars.perl < ${raw_data}/ntst14.fr-en.fr > ${data_dir}/test.fr
scripts/unescape-special-chars.perl < ${raw_data}/ntst14.fr-en.en > ${data_dir}/test.en

cur_dir=`pwd`
cd ${data_dir}

ln -s train.en train.char.en
ln -s train.fr train.char.fr
ln -s dev.en dev.char.en
ln -s dev.fr dev.char.fr
ln -s test.en test.char.en
ln -s test.fr test.char.fr

cd ${cur_dir}

scripts/prepare-data.py ${data_dir}/concat en fr ${data_dir} --mode vocab --character-level \
--vocab-prefix vocab.char --vocab-size 200
