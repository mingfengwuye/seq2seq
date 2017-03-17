#!/usr/bin/env bash

# share vocab & embeddings between MT and Edits
# target (edits) embedding has 2 parts:
# - word embedding, shared with MT
# - op type embedding

raw_data=experiments/APE16/raw_data
data_dir=experiments/APE16/edits/data_SUB_full

rm -rf ${data_dir}
mkdir -p ${data_dir}

cp ${raw_data}/{train,dev,test}.{src,mt,pe} ${data_dir}

scripts/extract-edits.py ${data_dir}/train.{mt,pe} --subs > ${data_dir}/train.edits
scripts/extract-edits.py ${data_dir}/dev.{mt,pe} --subs > ${data_dir}/dev.edits
scripts/extract-edits.py ${data_dir}/test.{mt,pe} --subs > ${data_dir}/test.edits

scripts/extract-edits.py ${data_dir}/train.{mt,pe} --subs --words | sed "s/<NONE>//g" > ${data_dir}/train.de

scripts/prepare-data.py ${data_dir}/train de ${data_dir} --no-tokenize --mode vocab --vocab-size 0
scripts/prepare-data.py ${data_dir}/train src ${data_dir} --no-tokenize --mode vocab --vocab-size 0
scripts/prepare-data.py ${data_dir}/train mt ${data_dir} --no-tokenize --mode vocab --vocab-size 0

head -n8 ${data_dir}/vocab.de > ${data_dir}/vocab.edits
tail -n+9 ${data_dir}/vocab.de | sed "s/^/<INS>_/" >> ${data_dir}/vocab.edits
tail -n+9 ${data_dir}/vocab.de | sed "s/^/<SUB>_/" >> ${data_dir}/vocab.edits

cat ${data_dir}/vocab.de > ${data_dir}/vocab_bis.mt
python3 -c "vocab = set(open('$data_dir/vocab.de')); print(''.join(line for line in open('$data_dir/vocab.mt') if line not in vocab))" >> ${data_dir}/vocab_bis.mt
mv ${data_dir}/vocab_bis.mt ${data_dir}/vocab.mt