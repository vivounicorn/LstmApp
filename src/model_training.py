#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from src.lstm_model import LstmModel
from src.data_processing import PoetrysDataSet


def train_lstm() -> None:
    cfg_file_path = '/home/zhanglei/Gitlab/LstmApp/config/cfg.ini'
    base_data = PoetrysDataSet(cfg_file_path)
    base_data.load_word2vec_model('../data//w2v_models/text_type_1.model')

    model = LstmModel(cfg_file_path, base_data, 'word2vec')
    # model.load('/home/zhanglei/Gitlab/LstmApp/data/models/model-270.hdf5')
    model.train_batch(mode='word2vec')
    # model.load('/home/zhanglei/Gitlab/LstmApp/data/models/model-270.hdf5')


if __name__ == '__main__':
    train_lstm()
