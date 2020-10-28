#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from configparser import ConfigParser
from src.constant import FILE_SECTION, PARAM_SECTION

class Config(object):

    def __init__(self, file_path):

        self.config = ConfigParser()
        if not os.path.exists(file_path):
            raise IOError("Can't read file(%s)" % file_path)

        self.config.read(file_path)
        if not self.config.has_section(FILE_SECTION):
            raise IOError("Can't read section(%s)" % FILE_SECTION)
        if not self.config.has_section(PARAM_SECTION):
            raise IOError("Can't read section(%s)" % PARAM_SECTION)

    def poetry_file_path(self):
        if self.config.has_option(FILE_SECTION, 'poetry_file_path'):
            return self.config.get(FILE_SECTION, 'poetry_file_path')
        return None

    def model_log_path(self):
        if self.config.has_option(FILE_SECTION, 'model_log_path'):
            return self.config.get(FILE_SECTION, 'model_log_path')
        return None

    def data_log_path(self):
        if self.config.has_option(FILE_SECTION, 'data_log_path'):
            return self.config.get(FILE_SECTION, 'data_log_path')
        return None

    def check_point_file_path(self):
        if self.config.has_option(FILE_SECTION, 'check_point_file_path'):
            return self.config.get(FILE_SECTION, 'check_point_file_path')
        return None

    def embedding_output_dim(self):
        if self.config.has_option(PARAM_SECTION, 'embedding_output_dim'):
            return self.config.getint(PARAM_SECTION, 'embedding_output_dim')
        return None

    def embedding_input_length(self):
        if self.config.has_option(PARAM_SECTION, 'embedding_input_length'):
            return self.config.getint(PARAM_SECTION, 'embedding_input_length')

    def vocab_size(self):
        if self.config.has_option(PARAM_SECTION, 'vocab_size'):
            return self.config.getint(PARAM_SECTION, 'vocab_size')
        return None

    def learning_rate(self):
        if self.config.has_option(PARAM_SECTION, 'learning_rate'):
            return self.config.getfloat(PARAM_SECTION, 'learning_rate')
        return None

    def batch_size(self):
        if self.config.has_option(PARAM_SECTION, 'batch_size'):
            return self.config.getint(PARAM_SECTION, 'batch_size')
        return None

    def lstm_layers_num(self):
        if self.config.has_option(PARAM_SECTION, 'lstm_layers_num'):
            return self.config.getint(PARAM_SECTION, 'lstm_layers_num')
        return None

    def dense_layers_num(self):
        if self.config.has_option(PARAM_SECTION, 'dense_layers_num'):
            return self.config.getint(PARAM_SECTION, 'dense_layers_num')
        return None

    def num_epochs(self):
        if self.config.has_option(PARAM_SECTION, 'num_epochs'):
            return self.config.getint(PARAM_SECTION, 'num_epochs')
        return None

    def valid_ratio(self):
        if self.config.has_option(PARAM_SECTION, 'valid_ratio'):
            return self.config.getfloat(PARAM_SECTION, 'valid_ratio')
        return None

    def test_ratio(self):
        if self.config.has_option(PARAM_SECTION, 'test_ratio'):
            return self.config.getfloat(PARAM_SECTION, 'test_ratio')
        return None
