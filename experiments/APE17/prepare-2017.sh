#!/usr/bin/env bash

raw_data=experiments/APE16/raw_data
data_dir=experiments/WMT17/data_2017

rm -rf ${data_dir}
mkdir -p ${data_dir}

cat ${raw_data}/{train,train.2017}.mt > ${data_dir}/train.mt
cat ${raw_data}/{train,train.2017}.pe > ${data_dir}/train.pe
cat ${raw_data}/{train,train.2017}.src > ${data_dir}/train.src
cp ${raw_data}/{dev,test,500K}.{src,mt,pe} ${data_dir}
cp ${raw_data}/test.2017.{src,mt} ${data_dir}

scripts/extract-edits.py ${data_dir}/train.{mt,pe} > ${data_dir}/train.edits
scripts/extract-edits.py ${data_dir}/dev.{mt,pe} > ${data_dir}/dev.edits
scripts/extract-edits.py ${data_dir}/test.{mt,pe} > ${data_dir}/test.edits
scripts/extract-edits.py ${data_dir}/500K.{mt,pe} > ${data_dir}/500K.edits

cp ${data_dir}/500K.mt ${data_dir}/train.concat.mt
cp ${data_dir}/500K.pe ${data_dir}/train.concat.pe
cp ${data_dir}/500K.src ${data_dir}/train.concat.src
cp ${data_dir}/500K.edits ${data_dir}/train.concat.edits

for i in {1..10}; do   # oversample PE data
    cat ${data_dir}/train.mt >> ${data_dir}/train.concat.mt
    cat ${data_dir}/train.pe >> ${data_dir}/train.concat.pe
    cat ${data_dir}/train.src >> ${data_dir}/train.concat.src
    cat ${data_dir}/train.edits >> ${data_dir}/train.concat.edits
done

cat ${data_dir}/train.{mt,pe} > ${data_dir}/train.de
cat ${data_dir}/train.concat.{mt,pe} > ${data_dir}/train.concat.de

# prepare subwords

scripts/learn_bpe.py -i ${data_dir}/train.de -o ${data_dir}/bpe.de -s 30000 --min-freq 5
scripts/learn_bpe.py -i ${data_dir}/train.src -o ${data_dir}/bpe.src -s 30000 --min-freq 5

scripts/apply_bpe.py -i ${data_dir}/train.de -o ${data_dir}/train.subwords.de -c ${data_dir}/bpe.de
for corpus in train dev test; do
    scripts/apply_bpe.py -i ${data_dir}/${corpus}.mt -o ${data_dir}/${corpus}.subwords.mt -c ${data_dir}/bpe.de
    scripts/apply_bpe.py -i ${data_dir}/${corpus}.pe -o ${data_dir}/${corpus}.subwords.pe -c ${data_dir}/bpe.de
    scripts/apply_bpe.py -i ${data_dir}/${corpus}.src -o ${data_dir}/${corpus}.subwords.src -c ${data_dir}/bpe.src
done
scripts/apply_bpe.py -i ${data_dir}/test.2017.mt -o ${data_dir}/test.2017.subwords.mt -c ${data_dir}/bpe.de
scripts/apply_bpe.py -i ${data_dir}/test.2017.src -o ${data_dir}/test.2017.subwords.src -c ${data_dir}/bpe.src

scripts/learn_bpe.py -i ${data_dir}/train.concat.de -o ${data_dir}/bpe.concat.de -s 30000 --min-freq 5
scripts/learn_bpe.py -i ${data_dir}/train.concat.src -o ${data_dir}/bpe.concat.src -s 30000 --min-freq 5

scripts/apply_bpe.py -i ${data_dir}/train.concat.de -o ${data_dir}/train.concat.subwords.de -c ${data_dir}/bpe.de
scripts/apply_bpe.py -i ${data_dir}/train.concat.mt -o ${data_dir}/train.concat.subwords.mt -c ${data_dir}/bpe.de
scripts/apply_bpe.py -i ${data_dir}/train.concat.pe -o ${data_dir}/train.concat.subwords.pe -c ${data_dir}/bpe.de
scripts/apply_bpe.py -i ${data_dir}/train.concat.src -o ${data_dir}/train.concat.subwords.src -c ${data_dir}/bpe.src
for corpus in dev test; do
    scripts/apply_bpe.py -i ${data_dir}/${corpus}.mt -o ${data_dir}/${corpus}.concat.subwords.mt -c ${data_dir}/bpe.de
    scripts/apply_bpe.py -i ${data_dir}/${corpus}.pe -o ${data_dir}/${corpus}.concat.subwords.pe -c ${data_dir}/bpe.de
    scripts/apply_bpe.py -i ${data_dir}/${corpus}.src -o ${data_dir}/${corpus}.concat.subwords.src -c ${data_dir}/bpe.src
done
scripts/apply_bpe.py -i ${data_dir}/test.2017.mt -o ${data_dir}/test.2017.concat.subwords.mt -c ${data_dir}/bpe.de
scripts/apply_bpe.py -i ${data_dir}/test.2017.src -o ${data_dir}/test.2017.concat.subwords.src -c ${data_dir}/bpe.src

# prepare vocabs

scripts/prepare-data.py ${data_dir}/train src de edits ${data_dir} --mode vocab --vocab-size 0
scripts/prepare-data.py ${data_dir}/train.concat src de edits ${data_dir} --mode vocab --vocab-prefix vocab.concat \
--vocab-size 30000

scripts/prepare-data.py ${data_dir}/train.subwords src de ${data_dir} --vocab-prefix vocab.subwords \
--no-tokenize --vocab-size 0 --mode vocab

scripts/prepare-data.py ${data_dir}/train.concat.subwords src de ${data_dir} --vocab-prefix vocab.concat.subwords \
--no-tokenize --vocab-size 0 --mode vocab

for vocab in vocab vocab.concat vocab.subwords vocab.concat.subwords; do
    cp ${data_dir}/${vocab}.de ${data_dir}/${vocab}.mt
    cp ${data_dir}/${vocab}.de ${data_dir}/${vocab}.pe
done
