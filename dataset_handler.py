import numpy as np
from numpy.random import RandomState
import os.path


def load_data(encoder, loc='./', seed=1234):
    z = {}
    pos, neg = load_mpqa(loc=loc)

    labels = compute_labels(pos, neg)
    text, labels = shuffle_data(pos+neg, labels, seed=seed)
    z['text'] = text
    z['labels'] = labels
#     print("Type of text is {}".format(text))
    features = encoder.encode(text)

    return z, features


def load_mpqa(loc='./'):
    pos, neg = [], []
    with open(os.path.join(loc, 'mpqa.pos'), 'rb') as f:
        for line in f:
            text = line.strip()
            if len(text) > 0:
                pos.append(text)
    with open(os.path.join(loc, 'mpqa.neg'), 'rb') as f:
        for line in f:
            text = line.strip()
            if len(text) > 0:
                neg.append(text)
    return pos, neg


def compute_labels(pos, neg):
    labels = np.zeros(len(pos) + len(neg))
    labels[:len(pos)] = 1.0
    labels[len(pos):] = 0.0
    return labels


def shuffle_data(X, L, seed=1234):
    prng = RandomState(seed)
    inds = np.arange(len(X))
    prng.shuffle(inds)
    X = [X[i] for i in inds]
    L = L[inds]
    return (X, L)    




