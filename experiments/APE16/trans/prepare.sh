#!/usr/bin/env bash

data_dir=experiments/APE16/trans/data_bis
raw_data=experiments/APE16/raw_data

rm -rf ${data_dir}
mkdir -p ${data_dir}

# cat ${raw_data}/{4M,500K,train}.mt > ${data_dir}/raw.mt
# cat ${raw_data}/{4M,500K,train}.src > ${data_dir}/raw.src
# cat ${raw_data}/{4M,500K,train}.pe > ${data_dir}/raw.pe

# cat ${raw_data}/train.src > ${data_dir}/raw.src
# cat ${raw_data}/train.mt > ${data_dir}/raw.de
# cat ${raw_data}/train.pe >> ${data_dir}/raw.de

cat ${raw_data}/{500K,train}.src > ${data_dir}/raw.src
cat ${raw_data}/{500K,train}.mt > ${data_dir}/raw.de
cat ${raw_data}/{500K,train}.pe >> ${data_dir}/raw.de

scripts/train-truecaser.perl --model ${data_dir}/truecaser.src --corpus ${data_dir}/raw.src
scripts/train-truecaser.perl --model ${data_dir}/truecaser.de --corpus ${data_dir}/raw.de

scripts/truecase.perl --model ${data_dir}/truecaser.src < ${data_dir}/raw.src > ${data_dir}/raw.true.src 2>/dev/null
scripts/truecase.perl --model ${data_dir}/truecaser.de < ${data_dir}/raw.de > ${data_dir}/raw.true.de 2>/dev/null

scripts/prepare-data.py ${data_dir}/raw.true src de ${data_dir} --subwords --vocab-size 40000 --no-tokenize --output trash
rm -f ${data_dir}/trash.{mt,de}

cp ${data_dir}/bpe.de ${data_dir}/bpe.mt
cp ${data_dir}/bpe.de ${data_dir}/bpe.pe

# add subword units that are not already included
scripts/concat-bpe.py ${data_dir}/vocab.de ${data_dir}/bpe.de > ${data_dir}/vocab_extended.de
mv ${data_dir}/vocab_extended.de ${data_dir}/vocab.de
scripts/concat-bpe.py ${data_dir}/vocab.src ${data_dir}/bpe.src > ${data_dir}/vocab_extended.src
mv ${data_dir}/vocab_extended.src ${data_dir}/vocab.src

cp ${data_dir}/vocab.de ${data_dir}/vocab.mt
cp ${data_dir}/vocab.de ${data_dir}/vocab.pe

scripts/truecase.perl --model ${data_dir}/truecaser.src < ${raw_data}/4M.src > ${data_dir}/raw.src 2>/dev/null
scripts/truecase.perl --model ${data_dir}/truecaser.de < ${raw_data}/4M.mt > ${data_dir}/raw.mt 2>/dev/null
scripts/truecase.perl --model ${data_dir}/truecaser.de < ${raw_data}/4M.pe > ${data_dir}/raw.pe 2>/dev/null

scripts/prepare-data.py ${data_dir}/raw src mt pe ${data_dir} --mode prepare --output pretrain --no-tokenize --bpe-path ${data_dir}/bpe --subwords

cp ${raw_data}/500K.mt ${data_dir}/raw.mt
cp ${raw_data}/500K.src ${data_dir}/raw.src
cp ${raw_data}/500K.pe ${data_dir}/raw.pe

for i in {1..20}; do   # oversample PE data
    cat ${raw_data}/train.src >> ${data_dir}/raw.src
    cat ${raw_data}/train.mt >> ${data_dir}/raw.mt
    cat ${raw_data}/train.pe >> ${data_dir}/raw.pe
done

scripts/truecase.perl --model ${data_dir}/truecaser.src < ${data_dir}/raw.src > ${data_dir}/train.true.src 2>/dev/null
scripts/truecase.perl --model ${data_dir}/truecaser.de < ${data_dir}/raw.mt > ${data_dir}/train.true.mt 2>/dev/null
scripts/truecase.perl --model ${data_dir}/truecaser.de < ${data_dir}/raw.pe > ${data_dir}/train.true.pe 2>/dev/null

scripts/truecase.perl --model ${data_dir}/truecaser.src < ${raw_data}/dev.src > ${data_dir}/dev.true.src 2>/dev/null
scripts/truecase.perl --model ${data_dir}/truecaser.de < ${raw_data}/dev.mt > ${data_dir}/dev.true.mt 2>/dev/null
scripts/truecase.perl --model ${data_dir}/truecaser.de < ${raw_data}/dev.pe > ${data_dir}/dev.true.pe 2>/dev/null

scripts/truecase.perl --model ${data_dir}/truecaser.src < ${raw_data}/test.src > ${data_dir}/test.true.src 2>/dev/null
scripts/truecase.perl --model ${data_dir}/truecaser.de < ${raw_data}/test.mt > ${data_dir}/test.true.mt 2>/dev/null
scripts/truecase.perl --model ${data_dir}/truecaser.de < ${raw_data}/test.pe > ${data_dir}/test.true.pe 2>/dev/null

scripts/prepare-data.py ${data_dir}/train.true src mt pe ${data_dir} --mode prepare --no-tokenize --bpe-path ${data_dir}/bpe --subwords \
--dev-corpus ${data_dir}/dev.true --test-corpus ${data_dir}/test.true
rm -f ${data_dir}/{raw,raw.true,train.true,dev.true,test.true}.{mt,src,pe}