#!/usr/bin/env bash

# data_dir=/home/aberard/local/data/WMT16/mono
# unxz --stdout ${data_dir}/commoncrawl.de.xz | scripts/well-formed.py | head -n1000000000 > ${data_dir}/commoncrawl.de

url=http://web-language-models.s3-website-us-east-1.amazonaws.com/wmt16/deduped/de.xz
dest=~/local/data/WMT16/mono/commoncrawl.de
mosesdecoder=~/local/moses
truecaser=experiments/APE16/synthetic/data/truecaser.de
bpe_path=experiments/APE16/synthetic/data/bpe.de

# 1. remove ill-formed
# 2. normalize punctuation
# 3. tokenize
# 4. truecase
# 5. apply subwords

time wget -qO- ${url} | unxz --stdout | \
scripts/well-formed.py | \
scripts/normalize-punctuation.perl -l de 2>/dev/null | \
scripts/tokenizer.perl -l de 2>/dev/null | \
${mosesdecoder}/scripts/recaser/truecase.perl --model ${truecaser} 2>/dev/null | \
scripts/apply_bpe.py -c ${bpe_path} | \
head -n1000000000 > ${dest}

# 6. compute LM scores & sort
# 7. compute TER statistics & sort/filter