#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Const:

    class ConstError(TypeError): pass

    def __setattr__(self, name, value):
        if self.__dict__.has_key(name):
            raise self.ConstError("Can't rebind const(%s)" % name)
        self.__dict__[name] = value

    def __delattr__(self, name):
        if self.__dict__.has_key(name):
            raise self.ConstError("Can't unbind const(%s)" % name)
        raise NameError(name)

import sys
sys.modules[__name__] = Const( )

Const.SPACE = u' '
Const.ONE_HOT = 'one-hot'
Const.WORD2VEC = 'word2vec'
Const.TRAIN_SET = 'train'
Const.VALID_SET = 'valid'
Const.TEST_SET = 'test'
Const.FILE_SECTION = 'file_path'
Const.PARAM_SECTION = 'parameters'