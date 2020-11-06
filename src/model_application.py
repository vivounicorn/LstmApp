#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from src.lstm_model import LstmModel
from src.data_processing import PoetrysDataSet
from src.word2vec import Word2vecModel


def train_word2vec(base_data) -> None:
    w2v = Word2vecModel()
    w2v.train_vec()

    # test.
    a = w2v.most_similar(str(base_data.w2i('床')))
    for i in range(len(a)):
        print(base_data.i2w(a[i][0]), a[i][1])


def train_lstm(base_data, finetune=None, mode='word2vec'):

    model = LstmModel(cfg_file_path, base_data, mode)
    # fine tune.
    if finetune is not None:
        model.load(finetune)

    model.train_batch(mode=mode)

    return model


def test_lstm(base_data, sentence, model_path=None, mode='word2vec'):

    model = LstmModel(cfg_file_path, base_data, mode)
    if model_path is not None:
        model.load(model_path)

    return model.generate_poetry(sentence, mode=mode)


if __name__ == '__main__':
    cfg_file_path = '/home/zhanglei/Gitlab/LstmApp/config/cfg.ini'
    w2vmodel_path = '/home/zhanglei/Gitlab/LstmApp/data/w2v_models/w2v_size200_sg1_hs0_ns3.model'
    model_path = '/home/zhanglei/Gitlab/LstmApp/data/models/model-2117.hdf5'

    base_data = PoetrysDataSet(cfg_file_path)
    # train_word2vec(base_data)
    base_data.load_word2vec_model(w2vmodel_path)

    # train_lstm(base_data=base_data, finetune=model_path)

    sentence = '人问寒山道，'

    for i in range(10):
        print(test_lstm(base_data=base_data, sentence=sentence, model_path=model_path))
