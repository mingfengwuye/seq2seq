#!/usr/bin/env bash

# url=http://web-language-models.s3-website-us-east-1.amazonaws.com/wmt16/deduped/de.xz

url=http://web-language-models.s3-website-us-east-1.amazonaws.com/wmt16/deduped/en-new.xz
dest=~/local/data/WMT17/mono/commoncrawl.en
mosesdecoder=~/local/moses

# 1. remove ill-formed     (28s)
# 2. normalize punctuation (60s)
# 3. tokenize              (82s)

time wget -qO- ${url} | unxz --stdout | \
scripts/well-formed-new.py | \
scripts/normalize-punctuation.perl -l en | \
scripts/tokenizer.perl -l en 2>/dev/null | \
head -n1000000000 > ${dest}

# commoncrawl.en: 584M sentences
