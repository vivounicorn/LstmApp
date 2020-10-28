#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from keras.models import Input, Sequential, load_model, Model
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras import losses

from src.utils import Logger
from src.config import Config


class LstmModel(object):
    def __init__(self,
                 cfg_path='/home/zhanglei/Gitlab/LstmApp/config/cfg.ini'):

        cfg = Config(cfg_path)

        global log
        log = Logger(cfg.model_log_path())
        self.model = None
        self.vocab_size = cfg.vocab_size()
        self.embedding_input_length = cfg.embedding_input_length()
        self.embedding_output_dim = cfg.embedding_output_dim()
        self.checkpoint = ModelCheckpoint(filepath=cfg.check_point_file_path(),
                                          save_weights_only=False,
                                          verbose=1)
        self.batch_size = cfg.batch_size()
        self.epochs = cfg.num_epochs()

        self._build(cfg.lstm_layers_num(),cfg.dense_layers_num())

    def _build(self,
              lstm_layers_num,
              dense_layers_num):

        units = 100
        model = Sequential()
        model.add(Input(shape=(self.embedding_input_length, self.vocab_size)))
        # model.add(Embedding(output_dim=self.embedding_output_dim,
        #                     input_dim=self.vocab_size,
        #                     input_length=self.embedding_input_length))

        model.add(LSTM(units=512,
                       activation='relu',
                       return_sequences=True))

        for i in range(lstm_layers_num - 1):
            model.add(LSTM(units=units * (i + 1),
                           dropout=0.6,
                           activation='relu',
                           return_sequences=False))

        for i in range(dense_layers_num - 1):
            model.add(Dense(units=units * (i + 1),
                            activation='relu'))
            model.add(Dropout(0.6))

        model.add(Dense(units=self.vocab_size,  # 可以是one-hot的字典表，也可以是w2v的向量flaten
                        activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

        model.summary()
        plot_model(model, to_file='../model.png', show_shapes=True, expand_nested=True)
        self.model = model
        return model

    def load(self, model_path):

        if not os.path.exists(model_path):
            log.error("the path of model is not exist.")
        return load_model(model_path)

    def save(self, model_path):

        self.save(model_path)
        return model_path

    def train_batch(self,
                    data_processor):

        log.debug("begin training")
        log.debug("batch_size:{0},steps_per_epoch:{1},epochs:{2}"
                  .format(self.batch_size,
                          data_processor.data_size_train // self.batch_size,
                          self.epochs))

        self.model.fit_generator(generator=data_processor.next_batch(batch_size=self.batch_size,
                                                                     batch_type='train'),
                                 verbose=True,
                                 steps_per_epoch=data_processor.data_size_train // self.batch_size,
                                 epochs=self.epochs,
                                 validation_data=data_processor.next_batch(batch_size=self.batch_size,
                                                                           batch_type='valid'),
                                 validation_steps=data_processor.data_size_valid // self.batch_size,
                                 callbacks=[self.checkpoint])
        log.debug("end training")

    def build_model(self):
        '''建立模型'''
        print('building model')

        # 输入的dimension
        input_tensor = Input(shape=(self.embedding_input_length, self.vocab_size))
        # input_tensor = Embedding(output_dim=self.embedding_output_dim,
        #                          input_dim=self.vocab_size,
        #                          input_length=self.embedding_input_length)
        model = Sequential()
        # model.add(Embedding(output_dim=self.embedding_output_dim,
        #                     input_dim=self.vocab_size,
        #                     input_length=self.embedding_input_length))
        model.add(Input(shape=(self.embedding_input_length, self.vocab_size)))
        model.add(LSTM(512, return_sequences=True))
        model.add(Dropout(0.6))
        model.add(LSTM(256))
        model.add(Dropout(0.6))
        model.add(Dense(self.vocab_size, activation='softmax'))
        self.model = model
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()
        plot_model(self.model, to_file='///model2.png', show_shapes=True,
                   expand_nested=True)

# m = LstmModel(embedding_input_length=6, embedding_output_dim=100)
# m.build_model()
# m.build(2,2)
