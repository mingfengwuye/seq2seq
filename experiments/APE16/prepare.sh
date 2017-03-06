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

main_dir=experiments/APE16
mosesdecoder=~/local/moses
data_dir=${main_dir}/data_trans
rm -rf ${data_dir}
mkdir -p ${data_dir}

cat ${main_dir}/raw_data/{4M,500K,train}.mt > ${data_dir}/raw.mt
cat ${main_dir}/raw_data/{4M,500K,train}.src > ${data_dir}/raw.src
cat ${main_dir}/raw_data/{4M,500K,train}.pe > ${data_dir}/raw.pe

${mosesdecoder}/scripts/recaser/train-truecaser.perl --model ${data_dir}/truecaser.mt --corpus ${data_dir}/raw.mt
${mosesdecoder}/scripts/recaser/train-truecaser.perl --model ${data_dir}/truecaser.src --corpus ${data_dir}/raw.src
${mosesdecoder}/scripts/recaser/train-truecaser.perl --model ${data_dir}/truecaser.pe --corpus ${data_dir}/raw.pe

${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.mt < ${data_dir}/raw.mt > ${data_dir}/raw.true.mt 2>/dev/null
${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.src < ${data_dir}/raw.src > ${data_dir}/raw.true.src 2>/dev/null
${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.pe < ${data_dir}/raw.pe > ${data_dir}/raw.true.pe 2>/dev/null

scripts/prepare-data.py ${data_dir}/raw.true src mt pe ${data_dir} --subwords --vocab-size 40000 --no-tokenize --output trash
rm ${data_dir}/{raw.true,trash}.{mt,src,pe}

${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.mt < ${main_dir}/raw_data/4M.mt > ${data_dir}/train.true.mt 2>/dev/null
${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.src < ${main_dir}/raw_data/4M.src > ${data_dir}/train.true.src 2>/dev/null
${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.pe < ${main_dir}/raw_data/4M.pe > ${data_dir}/train.true.pe 2>/dev/null

scripts/prepare-data.py ${data_dir}/train.true src mt pe ${data_dir} --mode prepare --output pretrain --no-tokenize --bpe-path ${data_dir}/bpe --subwords
rm ${data_dir}/train.true.{mt,src,pe}

cp ${main_dir}/raw_data/500K.mt ${data_dir}/raw.mt
cp ${main_dir}/raw_data/500K.src ${data_dir}/raw.src
cp ${main_dir}/raw_data/500K.pe ${data_dir}/raw.pe

for i in {1..20}; do   # oversample training data
    cat ${main_dir}/raw_data/train.mt >> ${data_dir}/raw.mt
    cat ${main_dir}/raw_data/train.src >> ${data_dir}/raw.src
    cat ${main_dir}/raw_data/train.pe >> ${data_dir}/raw.pe
done

# TODO: escape special characters
# TODO: use same subword units for MT and PE

${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.mt < ${data_dir}/raw.mt > ${data_dir}/train.true.mt 2>/dev/null
${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.src < ${data_dir}/raw.src > ${data_dir}/train.true.src 2>/dev/null
${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.pe < ${data_dir}/raw.pe > ${data_dir}/train.true.pe 2>/dev/null

${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.mt < ${main_dir}/raw_data/dev.mt > ${data_dir}/dev.true.mt 2>/dev/null
${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.src < ${main_dir}/raw_data/dev.src > ${data_dir}/dev.true.src 2>/dev/null
${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.pe < ${main_dir}/raw_data/dev.pe > ${data_dir}/dev.true.pe 2>/dev/null

${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.mt < ${main_dir}/raw_data/test.mt > ${data_dir}/test.true.mt 2>/dev/null
${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.src < ${main_dir}/raw_data/test.src > ${data_dir}/test.true.src 2>/dev/null
${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.pe < ${main_dir}/raw_data/test.pe > ${data_dir}/test.true.pe 2>/dev/null

scripts/prepare-data.py ${data_dir}/train.true src mt pe ${data_dir} --mode prepare --no-tokenize --bpe-path ${data_dir}/bpe --subwords \
--dev-corpus ${data_dir}/dev.true --test-corpus ${data_dir}/test.true
rm ${data_dir}/{train,dev,test}.true.{mt,src,pe}