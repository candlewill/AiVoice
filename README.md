# Deep Voice 3

This is a tensorflow implementation of [DEEP VOICE 3: 2000-SPEAKER NEURAL TEXT-TO-SPEECH](https://arxiv.org/pdf/1710.07654.pdf). For now, we are just focusing on single speaker synthesis.


## Requirement

* Tensorflow >= 1.2
* Python >= 3.0


## Dataset

[The LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset)

## Pre-process

Download and unzip the LJ Speech Dataset. Run:

```
python prepro.py
```

Note: Make sure that we have unzipped the dataset into the same foler of `prepro.py`.

After this, we would get three new folders:

```
├── dones          [New]
├── mags           [New]
├── mels           [New]
├── metadata.csv
├── README
└── wavs
```

## Training

Training data is loaded from `./LJSpeech-1.0/metadata.csv`, `./LJSpeech-1.0/mels`, `./LJSpeech-1.0/dones`, `./LJSpeech-1.0/mags` as default. If we want to change the loading path, we could change the config in `class Hyperparams`.

To train the model, we use this command:

```
python train.py
```

## Pre-trained Model

Currently, we can not get good result. However, we still provide our pre-trained model in case someone is interested in it.

[Pre-trained Model](https://cnbj1.fds.api.xiaomi.com/tts/ExternalLink/Github/pre_trained_model.tar.gz).

Its attention figure is as follows:

![Image of attention](https://cnbj1.fds.api.xiaomi.com/tts/ExternalLink/Github/alignment.png)

All the attention figures generated at training are included in the pre-trained model zipped file.

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