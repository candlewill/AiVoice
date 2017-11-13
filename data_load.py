# -*- coding: utf-8 -*-

'''
By Yunchao He. yunchaohe@gmail.com
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from utils import *
import codecs
import re
import os
import unicodedata


def text_normalize(sent):
    '''Minimum text preprocessing'''

    def _strip_accents(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn')

    normalized = re.sub("[^ a-z']", "", _strip_accents(sent).lower())
    return normalized


def load_vocab():
    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"  # P: Padding E: End of Sentence
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for idx, char in enumerate(vocab)}
    return char2idx, idx2char


def load_train_data():
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    # Parse
    texts, mels, dones, mags = [], [], [], []
    metadata = os.path.join(hp.data, 'metadata.csv')
    for line in codecs.open(metadata, 'r', 'utf-8'):
        fname, _, sent = line.strip().split("|")
        sent = text_normalize(sent) + "E"  # text normalization, E: EOS
        if len(sent) <= hp.T_x:
            sent += "P" * (hp.T_x - len(sent))
            texts.append(np.array([char2idx[char] for char in sent], np.int32).tostring())
            mels.append(os.path.join(hp.data, "mels", fname + ".npy"))
            dones.append(os.path.join(hp.data, "dones", fname + ".npy"))
            mags.append(os.path.join(hp.data, "mags", fname + ".npy"))

    return texts[:128], mels[:128], dones[:128], mags[:128]


def load_test_data():
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    # Parse
    texts = []
    for line in codecs.open('test_sents.txt', 'r', 'utf-8'):
        sent = text_normalize(line).strip() + "E"  # text normalization, E: EOS
        if len(sent) <= hp.T_x:
            sent += "P" * (hp.T_x - len(sent))
            texts.append([char2idx[char] for char in sent])
    texts = np.array(texts, np.int32)
    return texts


def load_npy(filename):
    # The type of filename is "bytes"
    filename = filename.decode("utf-8")
    return np.load(filename)


def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        _texts, _mels, _dones, _mags = load_train_data()  # bytes

        # Calc total batch count
        num_batch = len(_texts) // hp.batch_size

        # Convert to string tensor
        texts = tf.convert_to_tensor(_texts)
        mels = tf.convert_to_tensor(_mels, dtype=tf.string)
        dones = tf.convert_to_tensor(_dones, dtype=tf.string)
        mags = tf.convert_to_tensor(_mags, dtype=tf.string)

        # Create Queues
        text, mel, done, mag = tf.train.slice_input_producer([texts, mels, dones, mags], shuffle=True)

        # Decoding.
        text = tf.decode_raw(text, tf.int32)  # (T_x,)
        mel = tf.py_func(load_npy, [mel], tf.float32)  # (T_y/r, n_mels*r)
        done = tf.py_func(load_npy, [done], tf.int32)  # (T_y,)
        mag = tf.py_func(load_npy, [mag], tf.float32)  # (T_y, 1+n_fft/2)

        # create batch queues
        texts, mels, dones, mags = tf.train.batch([text, mel, done, mag],
                                                  shapes=[(hp.T_x,), (hp.T_y // hp.r, hp.n_mels * hp.r),
                                                          (hp.T_y // hp.r,), (hp.T_y, 1 + hp.n_fft // 2)],
                                                  num_threads=32,
                                                  batch_size=hp.batch_size,
                                                  capacity=hp.batch_size * 32,
                                                  dynamic_pad=False)

    return texts, mels, dones, mags, num_batch


if __name__ == '__main__':
    texts, mels, dones, mags, num_batch = get_batch()
    print("OK")
    sv = tf.train.Supervisor(logdir=hp.logdir, save_model_secs=0)
    with sv.managed_session() as sess:
        while True:
            print("Start.................")
            py_texts = sess.run(texts)
            print(py_texts)
