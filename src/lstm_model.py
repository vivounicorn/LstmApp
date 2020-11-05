#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from keras.models import Input, Sequential, load_model, Model
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, LambdaCallback
import random
from src.constant import ONE_HOT, WORD2VEC

from src.utils import Logger
from src.config import Config


class LstmModel(object):
    """
    Lstm model class.
    """

    def __init__(self,
                 cfg_path='/home/zhanglei/Gitlab/LstmApp/config/cfg.ini',
                 dataset=None,
                 mode='one-hot'):
        """
        To initialize lstm model.
        :param cfg_path: the path of configuration file.
        :param dataset: the data set of model training.
        :param mode: one-hot or word2vec encoding.
        """

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
        self.data_sets = dataset
        self.mode = mode

        self._build(cfg.lstm_layers_num(), cfg.dense_layers_num())

    def _build(self,
               lstm_layers_num,
               dense_layers_num):
        """
        To build a lstm model with lstm layers and densse layers.
        :param lstm_layers_num: The number of lstm layers.
        :param dense_layers_num:The number of dense layers.
        :return: model.
        """

        units = 256
        model = Sequential()

        if self.mode == WORD2VEC:
            dim = self.data_sets.w2v_model.size
        elif self.mode == ONE_HOT:
            dim = self.vocab_size
        else:
            raise ValueError("mode must be word2vec or one-hot.")

        model.add(Input(shape=(self.embedding_input_length, dim)))

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

    def load(self, model_path) -> None:
        """
        To load trained model.
        :param model_path: the path of exist model.
        :return: None.
        """

        if not os.path.exists(model_path):
            log.error("the path of model is not exist.")

        self.model = load_model(model_path)

    def save(self, model_path):
        """
        To save  trained model.
        :param model_path: the path of model saving.
        :return: None.
        """

        self.save(model_path)
        return model_path

    def train_batch(self, mode='one-hot') -> None:
        """
        To train model with batch.
        :param mode: one-hot or word2vec encoding.
        :return: None.
        """

        log.debug("begin training")
        log.debug("batch_size:{0},steps_per_epoch:{1},epochs:{2},validation_steps{3}"
                  .format(self.batch_size,
                          self.data_sets.data_size_train // self.batch_size,
                          self.epochs,
                          self.data_sets.data_size_valid // self.batch_size))

        self.model.fit_generator(generator=self.data_sets.next_batch(batch_size=self.batch_size,
                                                                     batch_type='train',
                                                                     mode=mode),
                                 verbose=True,
                                 steps_per_epoch=self.data_sets.data_size_train // self.batch_size,
                                 epochs=self.epochs,
                                 validation_data=self.data_sets.next_batch(batch_size=self.batch_size,
                                                                           batch_type='valid',
                                                                           mode=mode),
                                 validation_steps=self.data_sets.data_size_valid // self.batch_size,
                                 callbacks=[
                                     self.checkpoint,
                                     LambdaCallback(on_epoch_end=self.export_checkpoint_results)
                                 ])
        log.debug("end training")

    def evaluate_batch(self, mode='one-hot'):
        """
        To evaluate model with batch.
        :param mode: onr-hot or word2vec encoding.
        :return: prediction result.
        """
        log.debug("begin testing")
        log.debug("batch_size:{0},test{1}".format(self.batch_size,
                                                  self.data_sets.data_size_test // self.batch_size))
        pred = self.model.evaluate_generator(generator=self.data_sets.next_batch(batch_size=self.batch_size,
                                                                                 batch_type='test',
                                                                                 mode=mode),
                                             verbose=True,
                                             steps=self.data_sets.data_size_test // self.batch_size)
        log.debug(pred)
        log.debug("end testing")
        return np.argmax(pred, axis=1)

    def predict_base(self, sentence, isword2idx=True, mode='one-hot'):
        """
        To predict model.
        :param sentence: a sentence.
        :param isword2idx: if isword2idx=True then sentence='床前明月光'
                           if isword2idx=False then sentence=[321, 4721, 400, 3814, 282, 4999]
        :param mode: one-hot or word2vec encoding.
        :return: predicted word's index list.
        """
        sentence_vector = self.data_sets.sentence2vec(sentence, isword2idx, mode)
        preds = self.model.predict(sentence_vector, verbose=1)[0]
        pro = np.random.choice(range(len(preds)), 1, p=preds)
        next_index = int(pro.squeeze())
        return [next_index]

    def export_checkpoint_results(self, epoch, logs):
        """
        To export checkpoint result randomly every 6 iterations.
        :param epoch: current epoch.
        :param logs: current logs.
        :return: None
        """

        if epoch % 6 != 0:
            return

        log.info("==================Epoch {0}, Loss {1}=====================".format(epoch, logs['loss']))
        for i in range(6):
            generate = self.predict_random(mode=self.mode)
            log.info(generate)
        log.info("==================End=====================")

    def predict_random(self, length=24, mode='one-hot'):
        """
        Randomly select the first line of a poem from the sample and generate wuyanjueju.
        :param length: How long are the generated sentences.
        :param mode: one-hot or word2vec encoding.
        :return: a sentence. sentence = "共题诗句遍，".
        """

        index = random.randint(0, self.data_sets.data_size_test)
        sentence = self.data_sets.poetrys_vector_test[index][: self.embedding_input_length]
        generate = self.predict_sen(sentence, length=length, mode=mode)
        return self.data_sets.idxlst2sentence(generate)

    def predict_sen(self, text, length=24, isword2idx=False, mode='one-hot'):
        """
        Using a text of length "embedding_input_length" to predict one sentence.
        :param isword2idx: if isword2idx=True then sentence='床前明月光'
                           if isword2idx=False then sentence=[321, 4721, 400, 3814, 282, 4999]
        :param text: a text of length "embedding_input_length". sentence=[321, 4721, 400, 3814, 282, 4999]
        :param length: How long are the generated sentences.
        :param mode: one-hot or word2vec encoding.
        :return: a generated sentence.
        """
        if not self.model:
            log.error("The model is not be trained or loaded.")
            return

        if len(text) < self.embedding_input_length:
            log.error('the length of text should not be less than {0}'.format(self.embedding_input_length))
            return

        sentence = text[-self.embedding_input_length:]
        generate = sentence

        sentence = sentence[:self.embedding_input_length]
        tmp = []
        for i in range(length - self.embedding_input_length):
            pared = self.predict_base(sentence, isword2idx, mode)
            tmp += pared
            sentence = sentence[1:] + pared

        generate += tmp
        return generate

    def generate_poetry(self, text, length=24, mode='one-hot'):
        """
        Using a text of length "embedding_input_length" to generate a poetry.
        :param text: a text of length "embedding_input_length". text="我见一片海，"
        :param length: How long are the generated sentences.
        :param mode: one-hot or word2vec encoding.
        :return: a poetrry. "我见一片海，薄年草古流。从安身上作，犹舞得转惭。"
        """
        sc = self.data_sets.sentence2idxlist(text)
        generate = self.predict_sen(sc, isword2idx=False, length=length, mode=mode)
        return self.data_sets.idxlst2sentence(generate)