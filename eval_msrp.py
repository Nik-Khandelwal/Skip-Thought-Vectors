import numpy as np

from collections import defaultdict
from nltk.tokenize import word_tokenize
from numpy.random import RandomState
import os.path
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score as f1


def evaluate(encoder, loc='./'):

    print ('Preparing data...')
    traintext, testtext, labels = load_data(loc)

    print ('Computing training skipthoughts...')
    trainA = encoder.encode(traintext[0])
    trainB = encoder.encode(traintext[1])

    C = 4

    print ('Computing testing skipthoughts...')
    testA = encoder.encode(testtext[0])
    testB = encoder.encode(testtext[1])

    
    train_features = np.c_[np.abs(trainA - trainB), trainA * trainB, feats(traintext[0], traintext[1])]
    test_features = np.c_[np.abs(testA - testB), testA * testB, feats(testtext[0], testtext[1])]


    print ('Evaluating...')
    clf = LogisticRegression(C=C)
    clf.fit(train_features, labels[0])
    yhat = clf.predict(test_features)

    print ('Test accuracy: ' , str(clf.score(test_features, labels[1])))
    print ('Test F1: ' , str(f1(labels[1], yhat)))


def load_data(loc='./'):
    trainloc = os.path.join(loc, 'msr_paraphrase_train.txt')
    testloc = os.path.join(loc, 'msr_paraphrase_test.txt')

    trainA, trainB, testA, testB = [],[],[],[]
    trainS, devS, testS = [],[],[]

    with open(trainloc, 'rb') as f:
        for line in f:
            text = line.strip().decode("utf-8").split('\t')
            trainA.append(' '.join(word_tokenize(text[3])))
            trainB.append(' '.join(word_tokenize(text[4])))
            trainS.append(text[0])
    with open(testloc, 'rb') as f:
        for line in f:
            text = line.strip().decode("utf-8").split('\t')
            testA.append(' '.join(word_tokenize(text[3])))
            testB.append(' '.join(word_tokenize(text[4])))
            testS.append(text[0])

    trainS = [int(s) for s in trainS[1:]]
    testS = [int(s) for s in testS[1:]]

    return [trainA[1:], trainB[1:]], [testA[1:], testB[1:]], [trainS, testS]


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def feats(A, B):
    tA = [t.split() for t in A]
    tB = [t.split() for t in B]
    
    nA = [[w for w in t if is_number(w)] for t in tA]
    nB = [[w for w in t if is_number(w)] for t in tB]

    features = np.zeros((len(A), 6))

    for i in range(len(A)):
        if set(nA[i]) == set(nB[i]):
            features[i,0] = 1.

    for i in range(len(A)):
        if set(nA[i]) == set(nB[i]) and len(nA[i]) > 0:
            features[i,1] = 1.

    for i in range(len(A)):
        if set(nA[i]) <= set(nB[i]) or set(nB[i]) <= set(nA[i]): 
            features[i,2] = 1.

    for i in range(len(A)):
        features[i,3] = 1.0 * len(set(tA[i]) & set(tB[i])) / len(set(tA[i]))

    for i in range(len(A)):
        features[i,4] = 1.0 * len(set(tA[i]) & set(tB[i])) / len(set(tB[i]))

    for i in range(len(A)):
        features[i,5] = 0.5 * ((1.0*len(tA[i]) / len(tB[i])) + (1.0*len(tB[i]) / len(tA[i])))

    return features