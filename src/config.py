#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from configparser import ConfigParser
from src.constant import FILE_SECTION, PARAM_SECTION, WORD2VEC


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

    def dump_dir(self):
        if self.config.has_option(FILE_SECTION, 'dump_dir'):
            return self.config.get(FILE_SECTION, 'dump_dir')
        return None

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

    def corpus_file(self):
        if self.config.has_option(WORD2VEC, 'corpus_file'):
            return self.config.get(WORD2VEC, 'corpus_file')
        return None

    def vec_out(self):
        if self.config.has_option(WORD2VEC, 'vec_out'):
            return self.config.get(WORD2VEC, 'vec_out')
        return None

    def window(self):
        if self.config.has_option(WORD2VEC, 'window'):
            return self.config.getint(WORD2VEC, 'window')
        return None

    def size(self):
        if self.config.has_option(WORD2VEC, 'size'):
            return self.config.getint(WORD2VEC, 'size')
        return None

    def sg(self):
        if self.config.has_option(WORD2VEC, 'sg'):
            return self.config.getint(WORD2VEC, 'sg')
        return None

    def hs(self):
        if self.config.has_option(WORD2VEC, 'hs'):
            return self.config.getint(WORD2VEC, 'hs')
        return None

    def negative(self):
        if self.config.has_option(WORD2VEC, 'negative'):
            return self.config.getint(WORD2VEC, 'negative')
        return None
