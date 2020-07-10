import _pickle as pkl
from collections import OrderedDict
import argparse


def build_dictionary(text):
    wordcount = {}
    for cc in text:
        words = cc.split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 0
            wordcount[w] += 1

    sorted_words = sorted(list(wordcount.keys()), key=lambda x: wordcount[x], reverse=True)

    worddict = OrderedDict()
    for idx, word in enumerate(sorted_words):
        worddict[word] = idx+2 # 0: <eos>, 1: <unk>

    return worddict, wordcount


def load_dictionary(loc='./book_dictionary_large.pkl'):
    with open(loc, 'rb') as f:
        worddict = pkl.load(f)
    return worddict


def save_dictionary(worddict, wordcount, loc='./book_dictionary_large.pkl'):
    with open(loc, 'wb') as f:
        pkl.dump(worddict, f)
        pkl.dump(wordcount, f)


def build_and_save_dictionary(text, source):
    save_loc = source+".pkl"
    try:
        cached = load_dictionary(save_loc)
        print("Using cached dictionary at {}".format(save_loc))
        return cached
    except:
        pass
    print("unable to load from cached, building fresh")
    worddict, wordcount = build_dictionary(text)
    print("Got {} unique words".format(len(worddict)))
    print("Saving dictionary at {}".format(save_loc))
    save_dictionary(worddict, wordcount, save_loc)
    return worddict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text_file", type=str)
    args = parser.parse_args()

    print("Extracting text from {}".format(args.text_file))
    text = open(args.text_file, "rt").readlines()
    print("Extracting dictionary..")
    worddict, wordcount = build_dictionary(text)

    out_file = args.text_file+".pkl"
    print("Got {} unique words. Saving to file {}".format(len(worddict), out_file))
    save_dictionary(worddict, wordcount, out_file)
    print("Done.")
