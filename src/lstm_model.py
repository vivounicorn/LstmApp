#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from keras.models import Input, Sequential, load_model, Model
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
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
        self.learning_rate = cfg.learning_rate()

        self._build(cfg.lstm_layers_num(), cfg.dense_layers_num())

    def _build(self,
               lstm_layers_num,
               dense_layers_num):

        units = 256
        model = Sequential()
        model.add(Input(shape=(self.embedding_input_length, self.vocab_size)))
        # model.add(Embedding(output_dim=self.embedding_output_dim,
        #                     input_dim=self.vocab_size,
        #                     input_length=self.embedding_input_length))

        model.add(LSTM(units=512,
                       return_sequences=True))
        model.add(Dropout(0.6))

        for i in range(lstm_layers_num - 1):
            model.add(LSTM(units=units * (i + 1),
                           return_sequences=False))
            model.add(Dropout(0.6))

        for i in range(dense_layers_num - 1):
            model.add(Dense(units=units * (i + 1)))
            model.add(Dropout(0.6))

        model.add(Dense(units=self.vocab_size,  # 可以是one-hot的字典表，也可以是w2v的向量flaten
                        activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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
        log.debug("batch_size:{0},steps_per_epoch:{1},epochs:{2},validation_steps{3}"
                  .format(self.batch_size,
                          data_processor.data_size_train // self.batch_size,
                          self.epochs,
                          data_processor.data_size_valid // self.batch_size))

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

    def test_batch(self, data_processor):
        log.debug("begin testing")
        log.debug("batch_size:{0},test{1}".format(self.batch_size,
                                                  data_processor.data_size_test // self.batch_size))
        pred = self.model.evaluate_generator(generator=data_processor.next_batch(batch_size=self.batch_size,
                                                                                batch_type='test'),
                                            verbose=True,
                                            steps=data_processor.data_size_test // self.batch_size)
        # print(pred)
        # print(len(pred))
        # print(np.argmax(pred, axis=1))
        log.debug("end testing")
        # return np.argmax(pred, axis=1)

    # def predict_base(self, sentence):
    #     if len(sentence) < self.embedding_input_length:
    #         log.error('sentence length is larger than embedding input length:{0}/{1}' % len(sentence),
    #                   self.embedding_input_length)
    #         return
    #
    #     sentence = sentence[-self.embedding_input_length:]
    #     x_pred = np.zeros((1, self.embedding_input_length, len(self.words)))
    #     for t, char in enumerate(sentence):
    #         x_pred[0, t, self.word2numF(char)] = 1.
    #     preds = self.model.predict(x_pred, verbose=0)[0]
    #     next_index = self.sample(preds, temperature=temperature)
    #     next_char = self.num2word[next_index]
    #
    #     return next_char
    #     pass

    def build_model(self):
        '''建立模型'''
        print('building model')

        # # 输入的dimension
        # input_tensor = Input(shape=(self.embedding_input_length, self.vocab_size))
        # # input_tensor = Embedding(output_dim=self.embedding_output_dim,
        # #                          input_dim=self.vocab_size,
        # #                          input_length=self.embedding_input_length)
        # model = Sequential()
        # # model.add(Embedding(output_dim=self.embedding_output_dim,
        # #                     input_dim=self.vocab_size,
        # #                     input_length=self.embedding_input_length))
        # model.add(Input(shape=(self.embedding_input_length, self.vocab_size)))
        # model.add(LSTM(512, return_sequences=True))
        # model.add(Dropout(0.6))
        # model.add(LSTM(256))
        # model.add(Dropout(0.6))
        # model.add(Dense(self.vocab_size, activation='softmax'))
        # self.model = model
        # self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # self.model.summary()

        input_tensor = Input(shape=(self.embedding_input_length, self.vocab_size))
        lstm = LSTM(512, return_sequences=True)(input_tensor)
        dropout = Dropout(0.6)(lstm)
        lstm = LSTM(256)(dropout)
        dropout = Dropout(0.6)(lstm)
        dense = Dense(self.vocab_size, activation='softmax')(dropout)
        self.model = Model(inputs=input_tensor, outputs=dense)
        optimizer = Adam(lr=self.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        plot_model(self.model, to_file='../model2.png', show_shapes=True,
                   expand_nested=True)

m = LstmModel('/home/zhanglei/Gitlab/LstmApp/config/cfg.ini')
m.build_model()
# m.build(2,2)
