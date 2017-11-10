# Deep Voice 3

This is a tensorflow implementation of [DEEP VOICE 3: 2000-SPEAKER NEURAL TEXT-TO-SPEECH](https://arxiv.org/pdf/1710.07654.pdf). For now, we are just focusing on single speaker synthesis.


## Requirement

* Tensorflow >= 1.3
* Python >= 3.0


## Dataset

[The LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset)

## Pre-process

Download and unzip the LJ Speech Dataset. Run:

```
python prepro.py
```

## Training



## File Description

  * hyperparams.py: hyper parameters
  * prepro.py: creates inputs and targets, i.e., mel spectrogram, magnitude, and dones.
  * data_load.py
  * utils.py: several custom operational functions.
  * modules.py: building blocks for the networks.
  * networks.py: encoder, decoder, and converter
  * train.py: train
  * synthesize.py: inference
  * test_sents.txt: some test sentences in the paper.

## Reference

Most of the code is borrowed from [Kyubyong/deepvoice3](https://github.com/Kyubyong/deepvoice3).