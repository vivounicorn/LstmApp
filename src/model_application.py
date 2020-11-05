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


def train_lstm(base_data):

    model = LstmModel(cfg_file_path, base_data, 'word2vec')
    # fine tune.
    # model.load('/home/zhanglei/Gitlab/LstmApp/data/models/model-2700.hdf5')
    model.train_batch(mode='word2vec')

    return model


def test_lstm(base_data, sentence, model=None):
    if model is None:
        model = LstmModel(cfg_file_path, base_data, 'word2vec')

    # model.load('/home/zhanglei/Gitlab/LstmApp/data/models/model-2041.hdf5')
    return model.generate_poetry(sentence, mode='word2vec')


if __name__ == '__main__':
    cfg_file_path = '/home/zhanglei/Gitlab/LstmApp/config/cfg.ini'
    base_data = PoetrysDataSet(cfg_file_path)
    train_word2vec(base_data)
    base_data.load_word2vec_model('../data//w2v_models/w2v_size200_sg1_hs0_ns3.model')

    model = train_lstm(base_data)

    print(test_lstm(base_data, "面朝大海看，"))
