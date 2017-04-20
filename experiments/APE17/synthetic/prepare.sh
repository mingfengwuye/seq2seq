#!/usr/bin/env bash

moses_dir=~/local/moses
data_dir=experiments/APE17/synthetic/data
raw_data=experiments/APE17/raw_data
corpus=~/local/data/WMT17/mono/commoncrawl.en

shuf -n 100000 ${corpus} > ${data_dir}/mono.en

${moses_dir}/bin/lmplz -o 3 < ${data_dir}/mono.en > ${data_dir}/mono.arpa
${moses_dir}/bin/lmplz -o 3 < ${raw_data}/train.pe > ${data_dir}/train.arpa

${moses_dir}/bin/build_binary ${data_dir}/mono.arpa ${data_dir}/mono.blm
${moses_dir}/bin/build_binary ${data_dir}/train.arpa ${data_dir}/train.blm

${moses_dir}/bin/query ${data_dir}/mono.blm < ${corpus} | grep -ao 'Total:.*OOV' | cut -d' ' -f2,2 > ${corpus}.scores1
${moses_dir}/bin/query ${data_dir}/train.blm < ${corpus} | grep -ao 'Total:.*OOV' | cut -d' ' -f2,2 > ${corpus}.scores2

mkdir ~/local/tmp/

# paste -d' ' ${corpus}.scores1 ${corpus}.scores2 ${corpus} | awk '{printf "%.4f ",-($1+$2)/(NF-2); for(i=3;i<NF;i++) printf "%s ",$i; printf "%s\n",$NF}' > ${corpus}.concat.scores
paste -d' ' ${corpus}.scores2 ${corpus} | awk '{printf "%.4f",-$1/(NF-1); $1=""; print $0}' > ${corpus}.concat.scores
LC_NUMERIC=en_US.UTF-8 sort -T ~/local/tmp -g ${corpus}.concat.scores | cut -d' ' -f2- | uniq | head 10000000 > ${corpus}.concat.sorted
