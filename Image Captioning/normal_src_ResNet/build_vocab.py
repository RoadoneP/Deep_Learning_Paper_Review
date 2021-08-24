import nltk
import pickle
from collections import Counter
import itertools

CAPTION_PATH = "../flickr8k/token.txt"
TRAIN_PATH = "../flickr8k/trainImages.txt"
VOCAB_PATH = "./data/flickr8k_vocab.pkl"
THRESHOLD = 1

class Flickr8k(object):
    def __init__(self, caption_path):

        with open(caption_path) as f:
            all_captions = f.read().splitlines()

        captions = {}
        for i, idcap in enumerate(all_captions):
            x = idcap.split('#')
            name, cap = x[0], "#".join(x[1:])[2:]
            if name not in captions:
                captions[name] = []
            captions[name].append(cap)

        self.captions = captions

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab():
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    f8k_train = Flickr8k(caption_path=CAPTION_PATH, split_path=TRAIN_PATH)
    
    for i, cap in f8k_train.captions.items():
        tokens = nltk.tokenize.word_tokenize(cap['caption'].lower())
        counter.update(tokens)
        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(f8k_train.captions)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= THRESHOLD]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main():
    vocab = build_vocab()
    vocab_path = VOCAB_PATH
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))

if __name__ == '__main__':
  main()