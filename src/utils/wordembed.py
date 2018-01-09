!/usr/bin/python

# -*- coding: utf8 -*-

# Lê Phước Nghĩa
# cài đặt thư viện gemsim: pip install gensim
#word to vector
# https://github.com/cltl/wsd-dynamic-sense-vector/blob/master/perform_wsd.py

#tài liệu tham khảo: https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/


import sys

# sys.stdout = codecs.getwriter('utf_8')(sys.stdout)

# sys.stdin = codecs.getreader('utf_8')(sys.stdin)

from gensim.models import Word2Vec
import os

#from pyvi.pyvi import ViTokenizer, ViPosTagger


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname), encoding='utf-8'):
                yield line.split()

class WordEmbedding(object):
    """docstring for w2vEn"""
    def __init__(self, sentences, size = 100, window = 10, min_count = 1):
        # super(w2v_En, self).__init__()
        self.sentences = sentences
        self.size = size
        self.window = window
        self.min_count = min_count
    def train(self): #train model
        mod = Word2Vec(self.sentences, self.size, self.window, self.min_count, workers = 4)
        return mod
    def word_embedding(self, word, mod): #thuc hien w2v tren don vi tu
        vect = mod.wv[word]
        return vect
    def sentence_embedding(self, emd_sentences, mod): # w2v tren don vi cau
        vecarray = []
        for word in emd_sentences:
            vec = self.word_embedding(word, mod)
            vecarray.append(vec)
        return vecarray
    def save_model(self, path, mod): #luu model
        mod.save(path)  #('/tmp/mymodel')
    def load_model(self, path): #load model
        mod = Word2Vec.load(path)
        return mod
    def train_online(self, mod, _sentences): # train tiep de model ton hoa
        mod.train(_sentences)

if __name__ == "__main__":
    dirfile = '../input/train/word_emb/en_test'
    sentences = MySentences(dirfile)
    we = WordEmbedding(sentences)
    mod = we.train()
    vec = we.word_embedding("said", mod)
    vec2 = we.word_embedding("say", mod)
    print("said ={0}".format(mod.wv.similar_by_word("said")))
    print("say ={0}".format(mod.wv.similar_by_word("say")))
    #vec2 = we.word_embedding("school", mod)
    # print(vec)
    # print(vec2)

