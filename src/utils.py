#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

import logging.handlers

logging.basicConfig(level=logging.INFO)


class Logger(logging.Logger):
    def __init__(self, filename=None):
        super(Logger, self).__init__(self)
        # 日志文件名
        if filename is None:
            filename = './logs/pt.log'
        self.filename = filename

        fh = logging.handlers.TimedRotatingFileHandler(self.filename, 'D', 1, 30)
        fh.suffix = "%Y%m%d-%H%M.log"
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter('[%(asctime)s] - %(filename)s [Line:%(lineno)d] - [%(levelname)s]-[thread:%('
                                      'thread)s]-[process:%(process)s] - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.addHandler(fh)
        self.addHandler(ch)