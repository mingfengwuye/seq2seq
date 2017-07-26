#!/usr/bin/env bash

cat "${1:-/dev/stdin}" | sed "s/@@ //g" | scripts/detruecase.perl | scripts/unescape-special-chars.perl
