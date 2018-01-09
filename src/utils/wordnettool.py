# import sys
# import string
# import subprocess
# import os
# from itertools import *
# from math import log
# from collections import defaultdict
from nltk.corpus import wordnet as wn

#====================================================
#   Các thao tác với wordnet
#====================================================
import os
import sys
import ast

class wordnettk:
    """docstring for wordnettk"""
    def __init__(self, word):
        self.word = word    #Tu nhap vao
        self.morphy = self._get_morphy(word) #hinh vi cua tu
        self.num_synset = 1 #len(self.get_list_synset()) # so luong synset
        self.list_synset = self._get_list_synset() #danh sach cac synset
        self.list_sensekey = self._get_sense_key() #cac sense key
    def _get_list_synset(self):
        if self.morphy != None:
            ss = wn.synsets(self.morphy)
        else:
            ss = wn.synsets(self.word)
        if len(ss) > 1:
            self.num_synset = len(ss)
        return [s.name() for s in ss] #[dog.n.01, frump.n.01, dog.n.03,...]

    # def get_num_synset ():
    #     return len(self.get_list_synset())
    def _get_morphy(self, word):
        subword = word
        if '-' in word:
            subword = word.replace('-','')
        mor = wn.morphy(word) if wn.morphy(word)!=None else wn.morphy(subword)

        return mor # dogs => dog
    # #   (wn.synset('dog.n.01').definition()

    def get_definition_sense(self, sense):
        return str(wn.synset(sense.lower()).definition()) # trả về gloss

    #   wn.synset('dog.n.01').examples()[0], có thể có nhìu ví dụ
    def get_examples_sense(self, sense):
        ex = wn.synset(sense).examples()
        s = ''
        for e in ex:
            s = str(e) + ";"
        return s

    def get_lemmas_sense(self, sense):
        lemmas = wn.synset(sense).lemmas()
        return [str(lemma.name()).lower() for lemma in lemmas]   #trả về các lemmas của 1 synset: dog.n.01 => ['dog', 'domestic_dog', 'Canis_familiaris']

    def _get_sense_key(self):
        list_sense_key = []
        if len(self.list_synset) != 0:
            for sense in self.list_synset:
                lem = self.get_lemmas_sense(sense)
                try:
                    index = lem.index(self.morphy)
                except ValueError:
                    index = 0
                _sense_key = str(wn.synset(sense).lemmas()[index].key())
                list_sense_key.append(_sense_key)
        return  list_sense_key#wn.synset('dog.n.1').lemmas()[0].key() =
        #trả về sensekey của lammas i: u'dog%1:05:00::'
