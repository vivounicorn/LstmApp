#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from src.lstm_model import LstmModel
from src.data_processing import PoetrysDataSet


def train_lstm():
    cfg_file_path = '/home/zhanglei/Gitlab/LstmApp/config/cfg.ini'
    base_data = PoetrysDataSet(cfg_file_path)
    model = LstmModel(cfg_file_path, base_data)
    model.load('/home/zhanglei/Gitlab/LstmApp/data/models/model-190.hdf5')
    model.train_batch()
    model.load('/home/zhanglei/Gitlab/LstmApp/data/models/model-479.hdf5')
    model.evaluate_batch()

if __name__ == '__main__':
    train_lstm()
