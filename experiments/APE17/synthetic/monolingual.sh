#!/usr/bin/env bash

# data_dir=/home/aberard/local/data/WMT16/mono
# unxz --stdout ${data_dir}/commoncrawl.de.xz | scripts/well-formed.py | head -n1000000000 > ${data_dir}/commoncrawl.de
# url=http://web-language-models.s3-website-us-east-1.amazonaws.com/wmt16/deduped/de.xz

url=http://web-language-models.s3-website-us-east-1.amazonaws.com/wmt16/deduped/en-new.xz
dest=~/local/data/WMT16/mono/commoncrawl.en
mosesdecoder=~/local/moses
truecaser=experiments/APE17/synthetic/data/truecaser.en
bpe_path=experiments/APE17/synthetic/data/bpe.en

# 1. remove ill-formed     (28s)
# 2. normalize punctuation (60s)
# 3. tokenize              (82s)
# 4. truecase              (209s)
# 5. apply subwords        (209s)

time wget -qO- ${url} | unxz --stdout | \
scripts/well-formed.py | \
scripts/normalize-punctuation.perl -l en | \
scripts/tokenizer.perl -l en 2>/dev/null | \
${mosesdecoder}/scripts/recaser/truecase.perl --model ${truecaser} 2>/dev/null | \
scripts/apply_bpe.py -c ${bpe_path} | \
head -n1000000000 > ${dest}

# 6. compute LM scores & sort
# 7. compute TER statistics & sort/filter