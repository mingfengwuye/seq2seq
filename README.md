# seq2seq
Attention-based sequence to sequence learning

## Dependencies

* [TensorFlow for Python 3](https://www.tensorflow.org/get_started/os_setup.html)
* YAML and Matplotlib modules for Python 3: `sudo apt-get install python3-yaml python3-matplotlib`


## How to use


Train a model (CONFIG is a YAML configuration file, such as `config/default.yaml`):

    python3 -m translate CONFIG --train -v 


Translate text using an existing model:

    python3 -m translate CONFIG --decode FILE_TO_TRANSLATE --output OUTPUT_FILE
or for interactive decoding:

    python3 -m translate CONFIG --decode


Example model:

    experiments/WMT14/download.sh    # download WMT14 data into data/raw
    experiments/WMT14/prepare.sh     # preprocess the data, and copy the files to experiments/WMT14/data
    python3 -m translate experiments/WMT14/baseline.yaml --train -v   # train a baseline model on this data


## Features
* **YAML configuration files**
* **Beam-search decoder**
* **External language model**
* **Ensemble decoding**
* **Multiple encoders**
* **Hierarchical encoder**
* **Bidirectional encoder**
* **Local attention model**
* **Convolutional attention model**
* **Detailed logging**
* **Periodic BLEU evaluation**
* **Periodic checkpoints**
* **Multi-task training:** train on several tasks at once (e.g. French->English and German->English MT)
* **Subwords training and decoding**
* **Input binary features instead of text**
* **Pre-processing script:** we provide a fully-featured Python script for data pre-processing (vocabulary creation, lowercasing, tokenizing, splitting, etc.)
* **Dynamic RNNs:** we use symbolic loops instead of statically unrolled RNNs. This means faster model creation, and that we don't need buckets

## Speech translation
* To replicate the results from [arxiv.org/abs/1612.01744](https://arxiv.org/abs/1612.01744), you should look at the branch `speech`. The default branch `baseline` implements a different model, from [arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473).

## Credits

* This project is based on [TensorFlow's reference implementation](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/models/rnn)
* We include some of the pre-processing scripts from [Moses](http://www.statmt.org/moses/)
* The scripts for subword units come from [github.com/rsennrich/subword-nmt](https://github.com/rsennrich/subword-nmt)
