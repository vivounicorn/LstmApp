#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing
import numpy as np

from src.config import Config
from src.utils import Logger


class Word2vecModel(object):
    """
    Word2vec model class.
    """
    def __init__(self,
                 cfg_path='/home/zhanglei/Gitlab/LstmApp/config/cfg.ini',
                 is_ns=False):
        """
        To initialize model.
        :param cfg_path: he path of configration file.
        :param model_type:
        """

        cfg = Config(cfg_path)
        global log
        log = Logger(cfg.model_log_path())
        self.model = None
        self.is_ns = is_ns
        self.vec_out = cfg.vec_out()
        self.corpus_file = cfg.corpus_file()
        self.window = cfg.window()
        self.size = cfg.size()
        self.sg = cfg.sg()
        self.hs = cfg.hs()
        self.negative = cfg.negative()

    def train_vec(self) -> None:
        """
        To train a word2vec model.
        :return: None
        """
        output_model = self.vec_out + 'w2v_size{0}_sg{1}_hs{2}_ns{3}.model'.format(self.size,
                                                                                   self.sg,
                                                                                   self.hs,
                                                                                   self.negative)

        output_vector = self.vec_out + 'w2v_size{0}_sg{1}_hs{2}_ns{3}.vector'.format(self.size,
                                                                                     self.sg,
                                                                                     self.hs,
                                                                                     self.negative)

        if not self.is_ns:
            self.model = Word2Vec(LineSentence(self.corpus_file),
                                  size=self.size,
                                  window=self.window,
                                  sg=self.sg,
                                  hs=self.hs,
                                  workers=multiprocessing.cpu_count())
        else:
            self.model = Word2Vec(LineSentence(self.corpus_file),
                                  size=self.size,
                                  window=self.window,
                                  sg=self.sg,
                                  hs=self.hs,
                                  negative=self.negative,
                                  workers=multiprocessing.cpu_count())

        self.model.save(output_model)
        self.model.wv.save_word2vec_format(output_vector, binary=False)

    def load(self, path):
        """
        To load a word2vec model.
        :param path: the model file path.
        :return: success True otherwise False.
        """
        try:
            self.model = gensim.models.Word2Vec.load(path)
            return True
        except:
            return False

    def most_similar(self, word):
        """
        Return the most similar words.
        :param word: a word.
        :return: similar word list.
        """
        word = self.model.most_similar(word)
        for text in word:
            log.info("word:{0} similar:{1}".format(text[0], text[1]))

        return word

    def get_vector(self, word):
        """
        To get a word's vector.
        :param word: a word.
        :return: word's word2vec vector.
        """
        try:
            return self.model.wv.get_vector(str(word))
        except KeyError:
            return np.zeros(
                shape=(self.size,),
                dtype=np.float
            )

    def get_embedding_layer(self, train_embeddings=False):
        """
        To get keras embedding layer from model.
        :param train_embeddings: if frozen the layer.
        :return: embedding layer.
        """
        try:
            return self.model.wv.get_keras_embedding(train_embeddings)
        except KeyError:
            return None
