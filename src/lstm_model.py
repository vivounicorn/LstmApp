#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from keras.models import Input, Sequential, load_model, Model
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.optimizers import Adam
from keras import losses

from src.utils import Logger
from src.config import Config


class LstmModel(object):
    def __init__(self,
                 cfg_path='/home/zhanglei/Gitlab/LstmApp/config/cfg.ini',
                 dataset=None):

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
        self.dataset = dataset

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

        self.model = load_model(model_path)

    def save(self, model_path):

        self.save(model_path)
        return model_path

    def train_batch(self):

        log.debug("begin training")
        log.debug("batch_size:{0},steps_per_epoch:{1},epochs:{2},validation_steps{3}"
                  .format(self.batch_size,
                          self.dataset.data_size_train // self.batch_size,
                          self.epochs,
                          self.dataset.data_size_valid // self.batch_size))

        self.model.fit_generator(generator=self.dataset.next_batch(batch_size=self.batch_size,
                                                                     batch_type='train'),
                                 verbose=True,
                                 steps_per_epoch=self.dataset.data_size_train // self.batch_size,
                                 epochs=self.epochs,
                                 validation_data=self.dataset.next_batch(batch_size=self.batch_size,
                                                                           batch_type='valid'),
                                 validation_steps=self.dataset.data_size_valid // self.batch_size,
                                 callbacks=[
                                     self.checkpoint,
                                     LambdaCallback(on_epoch_end=self.generate_sample_result)
                                 ])
        log.debug("end training")

    def evaluate_batch(self):
        log.debug("begin testing")
        log.debug("batch_size:{0},test{1}".format(self.batch_size,
                                                  self.dataset.data_size_test // self.batch_size))
        pred = self.model.evaluate_generator(generator=self.dataset.next_batch(batch_size=self.batch_size,
                                                                                batch_type='test'),
                                            verbose=True,
                                            steps=self.dataset.data_size_test // self.batch_size)
        log.debug(pred)
        # print(len(pred))
        # print(np.argmax(pred, axis=1))
        log.debug("end testing")
        return np.argmax(pred, axis=1)

    def sample(self, preds, temperature=1.0):
        '''
        当temperature=1.0时，模型输出正常
        当temperature=0.5时，模型输出比较open
        当temperature=1.5时，模型输出比较保守
        在训练的过程中可以看到temperature不同，结果也不同
        就是一个概率分布变换的问题，保守的时候概率大的值变得更大，选择的可能性也更大
        '''
        preds = np.asarray(preds).astype('float64')
        exp_preds = np.power(preds, 1. / temperature)
        preds = exp_preds / np.sum(exp_preds)
        pro = np.random.choice(range(len(preds)), 1, p=preds)
        return int(pro.squeeze())

    def predict_base(self, sentence, isword2idx=True):
        '''
        if isword2idx=True then sentence='床前明月光'
        if isword2idx=False then sentence=[321, 4721, 400, 3814, 282, 4999]
        return word's index
        '''
        sentence_vector = self.dataset.sentence2vec(sentence, isword2idx)
        preds = self.model.predict(sentence_vector, verbose=1)[0]
        log.debug(np.argmax(preds))
        next_index = self.sample(preds, temperature=1)
        # next_char = self.dataset.i2w(next_index)

        log.debug(next_index)
        return [next_index]

    #--------------------------------------------------------
    def generate_sample_result(self, epoch, log):
        '''训练过程中，每4个epoch打印出当前的学习情况'''
        if epoch % 6 != 0:
            return

        with open('../data/out.txt', 'a', encoding='utf-8') as f:
            f.write('==================Epoch {}=====================\n'.format(epoch))

        print("\n==================Epoch {}=====================".format(epoch))
        for diversity in [0.7, 1.0, 1.3]:
            print("------------Diversity {}--------------".format(diversity))
            generate = self.predict_random()
            print(generate)

            # 训练时的预测结果写入txt
            with open('../data/out.txt', 'a', encoding='utf-8') as f:
                f.write(generate + '\n')

    def predict_random(self):
        '''随机从库中选取一句开头的诗句，生成五言绝句
        sentence = [1921, 2108, 318, 577, 804, 4999]'''
        if not self.model:
            print('model not loaded')
            return
        import random

        index = random.randint(0, self.dataset.data_size_train)
        sentence = self.dataset.poetrys_vector_train[index][: self.embedding_input_length]
        generate = self.predict_sen(sentence)
        return self.dataset.idxlst2sentence(generate)


    def _preds(self, sentence, length=23):
        '''
        sentence:预测输入值
        lenth:预测出的字符串长度
        供类内部调用，输入max_len长度字符串，返回length长度的预测值字符串
        sentence=[321, 4721, 400, 3814, 282, 4999]
        '''
        sentence = sentence[:self.embedding_input_length]
        generate = []
        for i in range(length):
            pred = self.predict_base(sentence, False)
            generate += pred
            print('pred:',pred)
            print('sen:',sentence)
            sentence = sentence[1:] + pred
        return generate


    def predict_sen(self, text):
        '''根据给出的前max_len个字，生成诗句'''
        '''text=[321, 4721, 400, 3814, 282, 4999]'''
        '''此例中，即根据给出的第一句诗句（含逗号），来生成古诗'''
        if not self.model:
            return
        max_len = self.embedding_input_length
        if len(text) < max_len:
            print('length should not be less than ', max_len)
            return

        sentence = text[-max_len:]
        print('the first line:', sentence)
        generate = sentence
        generate += self._preds(sentence, 24-self.embedding_input_length)
        return generate

    def gen_poetry(self, seed_text, rows=4, cols=5):
        '''
        生成詩詞的函式
        輸入: 兩個漢字, 行數, 每行的字數 (預設為五言絕句)
        '''
        total_cols = cols + 1  # 加上標點符號
        import re, random
        chars = re.findall('[\x80-\xff]{3}|[\w\W]', seed_text)
        if len(chars) != self.embedding_input_length:  # seq_len = 2
            return ""
        #
        arr = [self.dataset.word2idx[k] for k in chars]
        for i in range(self.embedding_input_length, rows * total_cols):
            if (i + 1) % total_cols == 0:  # 逗號或句號
                if (i + 1) / total_cols == 2 or (i + 1) / total_cols == 4:  # 句號的情況
                    arr.append(2)  # 句號在字典中的對映為 2
                else:
                    arr.append(1)  # 逗號在字典中的對映為 1
            else:
                sentence_vector = self.dataset.sentence2vec(seed_text)
                proba = self.model.predict(np.array(arr[-self.embedding_input_length:]), verbose=0)
                predicted = np.argsort(proba[1])[-5:]
                index = random.randint(0, len(predicted) - 1)  # 在前五個可能結果裡隨機取, 避免每次都是同樣的結果
                new_char = predicted[index]
                while new_char == 1 or new_char == 2:  # 如果是逗號或句號, 應該重新換一個
                    index = random.randint(0, len(predicted) - 1)
                    new_char = predicted[index]
                arr.append(new_char)
        poem = [self.dataset.idx2word[i] for i in arr]
        return "".join(poem)

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
