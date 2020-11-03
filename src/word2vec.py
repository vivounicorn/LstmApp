#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing
import numpy as  np

from src.config import Config
from src.utils import Logger


class Word2vecModel(object):
    def __init__(self,
                 cfg_path='/home/zhanglei/Gitlab/LstmApp/config/cfg.ini',
                 model_type=1):

        cfg = Config(cfg_path)
        global log
        log = Logger(cfg.model_log_path())
        self.model = None
        self.model_type = model_type
        self.vec_out = cfg.vec_out()
        self.corpus_file = cfg.corpus_file()
        self.window = cfg.window()
        self.size = cfg.size()
        self.sg = cfg.sg()
        self.hs = cfg.hs()
        self.negative = cfg.negative()

    def train_vec(self):
        outp_model = self.vec_out + 'text_type_{}.model'.format(self.model_type)
        outp_vec = self.vec_out + 'text_type_{}.vector'.format(self.model_type)

        # 选择训练的方式
        if self.model_type == 1:
            # 1.hs-SkipGram
            self.model = Word2Vec(LineSentence(self.corpus_file),
                                  size=self.size,
                                  window=self.window,
                                  sg=self.sg,
                                  hs=self.hs,
                                  workers=multiprocessing.cpu_count())
        elif self.model_type == 2:
            # 2.负采样-CBOW
            self.model = Word2Vec(LineSentence(self.corpus_file),
                                  size=self.size,
                                  window=self.window,
                                  sg=self.sg,
                                  hs=self.hs,
                                  negative=self.negative,
                                  workers=multiprocessing.cpu_count())
        elif self.model_type == 3:
            # 3.负采样-SkipGram
            self.model = Word2Vec(LineSentence(self.corpus_file),
                                  size=self.size,
                                  window=self.window,
                                  sg=self.sg,
                                  hs=self.hs,
                                  negative=self.negative,
                                  workers=multiprocessing.cpu_count())
        # 保存模型
        self.model.save(outp_model)
        self.model.wv.save_word2vec_format(outp_vec, binary=False)

    def load(self, path):
        try:
            self.model = gensim.models.Word2Vec.load(path)
            return True
        except:
            return False

    def most_similar(self, word):
        word = self.model.most_similar(word)
        for text in word:
            print(text[0], text[1])

        return word

    def get_vector(self, word):
        try:
            return self.model.wv.get_vector(str(word))
        except KeyError:
            log.error("key of w2v model is not exist.{0}".format(word))
            return np.zeros(
                shape=(self.size,),
                dtype=np.float
            )

    def get_embedding_layer(self, train_embeddings=False):
        try:
            return self.model.wv.get_keras_embedding(train_embeddings)
        except KeyError:
            log.error("key of w2v model is not exist.")
            return None
