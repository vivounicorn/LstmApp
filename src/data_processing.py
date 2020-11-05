#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import numpy as np
from collections import Counter
import operator
from functools import reduce

from sklearn.model_selection import train_test_split
from src.utils import Logger
from src.constant import SPACE, ONE_HOT, WORD2VEC, TRAIN_SET, TEST_SET, VALID_SET
from src.config import Config
from src.word2vec import Word2vecModel


class PoetrysDataSet(object):
    """
    The poetrys data processor class.
     to analysis data and generate relevant data structures.
    """
    def __init__(self,
                 cfg_path='home/zhanglei/Gitlab/LstmApp/config/cfg.ini',
                 vocab=None,
                 word2idx=None,
                 idx2word=None):
        """
        To initialize poetrys data processing class.
        :param cfg_path: the path of configration file.
        :param vocab: vocabulary of data set.
        :param word2idx: mappping of word to index.
        :param idx2word: mapping of index to word.
        """

        cfg = Config(cfg_path)
        global log
        log = Logger(cfg.data_log_path())
        self.poetrys = []  # ['孤', '峰', '去', '，', '灰', '飞', '一', '烬',],['休', '。', '云', '无', '空', '碧', '在', '，', '天'],...
        self.vocab = set()  # ['丁', '七', '万', '丈', '三', '上', '下', '不', '与', '丐', '丑', '专', '且', '丕', '世', '丘', '丙', ...]
        self.word2idx = {}  # {' ': 0, '2': 1, '3': 2, '6': 3, '7': 4, '8': 5, ';': 6, 'F': 7, 'p': 8, 'í': 9, 'ó': 10,...}
        self.idx2word = {}  # {0: ' ', 1: '2', 2: '3', 3: '6', 4: '7', 5: '8', 6: ';', 7: 'F', 8: 'p', 9: 'í', 10: 'ó', ...}
        self.poetrys_vector = []  # [[1355, 6755, 4305, 1731, 658, 7444, 2405], [6290, 7272, 1104, 1665, 21, 478], [6952, 6961, 1580, 2626, 7444, 24]]
        self.poetrys_vector_train = []  # training vectors
        self.poetrys_vector_valid = []  # validating vectors
        self.poetrys_vector_test = []  # testing vectors

        self.all_words = [SPACE]  # char of space.

        self.w2v_model = None  # the word2vec model that has been trained.

        self._data_size = 0  # size of the full sample set.
        self.data_size_test = 0  # size of the testing set.
        self.data_size_train = 0  # size of the training set.
        self.data_size_valid = 0  # size of the validation set

        self._valid_size = cfg.valid_ratio()
        self._test_size = cfg.test_ratio()

        self.vocab_size = cfg.vocab_size()
        self.dump_dir = cfg.dump_dir()

        self.embedding_input_length = cfg.embedding_input_length()
        self._build_base(cfg.poetry_file_path(),
                         vocab,
                         word2idx,
                         idx2word)

        self._train_valid_test_split()

    def _build_base(self,
                    file_path,
                    vocab=None,
                    word2idx=None,
                    idx2word=None) -> None:
        """
        To scan the file and build vocabulary and so on.
        :param file_path: the file path of poetic corpus, one poem per line.
        :param vocab: the vocabulary.
        :param word2idx: the mapping of word to index.
        :param idx2word: the mapping of index to word
        :return: None.
        """

        pattern = re.compile(u"_|\(|（|《")
        with open(file_path, "r", encoding='UTF-8') as f:
            for line in f:
                try:
                    line = line.strip(u'\n')
                    title, content = line.strip(SPACE).split(u':')
                    content = content.replace(SPACE, u'')
                    idx = re.search(pattern, content)
                    if idx is not None:
                        content = content[:idx.span()[0]]

                    if len(content) < self.embedding_input_length:  # Filter data according to embedding input
                        # length to improve accuracy.
                        continue

                    words = []
                    for i in range(0, len(content)):
                        word = content[i:i + 1]
                        if (i + 1) % self.embedding_input_length == 0 and word not in [',', ',', '，', '.', '。']:
                            words = []
                            break
                        words.append(word)
                        self.all_words.append(word)

                    if len(words) > 0:
                        self.poetrys.append(words)

                except Exception as e:
                    log.error(str(e))

        if vocab is None:
            top_n = Counter(self.all_words).most_common(self.vocab_size - 1)
            top_n.append(SPACE)
            self.vocab = sorted(set([i[0] for i in top_n]))
        else:
            top_n = list(vocab)[:self.vocab_size - 1]
            top_n.append(SPACE)
            self.vocab = sorted(set([i for i in top_n]))  # cut vocab with threshold.

        log.debug(self.vocab)

        if word2idx is None:
            self.word2idx = dict((c, i) for i, c in enumerate(self.vocab))
        else:
            self.word2idx = word2idx

        if idx2word is None:
            self.idx2word = dict((i, c) for i, c in enumerate(self.vocab))
        else:
            self.idx2word = idx2word

        # Function of mapping word to index.
        self.w2i = lambda word: self.word2idx.get(str(word)) if self.word2idx.get(word) is not None \
            else self.word2idx.get(SPACE)
        # Function of mapping index to word.
        self.i2w = lambda idx: self.idx2word.get(int(idx)) if self.idx2word.get(int(idx)) is not None \
            else SPACE

        # Full vectors.
        self.poetrys_vector = [list(map(self.w2i, poetry)) for poetry in self.poetrys]
        self._data_size = len(self.poetrys_vector)
        self._data_index = np.arange(self._data_size)
        log.debug((self.poetrys_vector[0:2]))
        log.debug((self.poetrys[0:2]))

    def _dump_list(self, filename, memory_list) -> None:
        """
        To dump the list structure.
        :param filename: path for saving file.
        :param memory_list: list structure.
        :return: None
        """
        with open(filename, 'w') as f:
            for i in range(0, len(memory_list)):
                f.write(' '.join([str(item) for item in memory_list[i]]) + "\n")

    def _dump_dict(self, filename, memory_dict) -> None:
        """
        To dump the dictionary structure.
        :param filename: path for saving file.
        :param memory_dict: dictionary structure
        :return: None
        """
        with open(filename, 'w') as f:
            for i, fea in enumerate(memory_dict):
                f.write(' '.join([str(fea), str(memory_dict[fea])]) + "\n")

    def _print_vector(self, vec):
        """
        To print all elements of the vector.
        :param vec: vector
        :return: None
        """
        out = []
        for v in vec:
            if len(v.shape) == 2:
                for i in range(0, v.shape[0]):
                    for j in range(0, v.shape[1]):
                        if int(v[i][j]) != 0:
                            out.append(self.idx2word[j])
            elif len(v.shape) == 1:
                for i in range(0, v.shape[0]):
                    if int(v[i]) != 0:
                        out.append(self.idx2word[i])
        return out

    def _one_hot_encoding(self, sample):
        """
        One-hot encoding for a sample, a sample will be split into multiple samples.
        :param sample: a sample. [1257, 6219, 3946]
        :return: feature and label. feature:[[0,0,0,1,0,0,......],
                                            [0,0,0,0,0,1,......],
                                            [1,0,0,0,0,0,......]];
                                    label:  [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0......]
        """
        if type(sample) != list or 0 == len(sample):
            log.error("type or length of sample is invalid.")
            return None, None
        feature_samples = []
        label_samples = []
        idx = 0
        while idx < len(sample) - self.embedding_input_length:
            feature = sample[idx: idx + self.embedding_input_length]
            label = sample[idx + self.embedding_input_length]

            label_vector = np.zeros(
                shape=(1, self.vocab_size),
                dtype=np.float
            )
            label_vector[0, label] = 1.0

            feature_vector = np.zeros(
                shape=(1, self.embedding_input_length, self.vocab_size),
                dtype=np.float
            )

            for i, f in enumerate(feature):
                feature_vector[0, i, f] = 1.0

            idx += 1
            feature_samples.append(feature_vector)
            label_samples.append(label_vector)

        return feature_samples, label_samples

    def _word2vec_encoding(self, sample):
        """
        word2vec encoding for sample, a sample will be split into multiple samples.
        :param sample: a sample. [1257, 6219, 3946]
        :return: feature and label.feature:[[0.01,0.23,0.05,0.1,0.33,0.25,......],
                                            [0.23,0.45,0.66,0.32,0.11,1.03,......],
                                            [1.22,0.99,0.68,0.7,0.8,0.001,......]];
                                    label:  [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0......]
        """
        if type(sample) != list or 0 == len(sample):
            log.error("type or length of sample is invalid.")
            return None, None
        feature_samples = []
        label_samples = []
        idx = 0
        while idx < len(sample) - self.embedding_input_length:
            feature = sample[idx: idx + self.embedding_input_length]
            label = sample[idx + self.embedding_input_length]

            if self.w2v_model is None:
                log.error("word2vec model is none.")
                return None, None

            label_vector = np.zeros(
                shape=(1, self.vocab_size),
                dtype=np.float
            )
            label_vector[0, label] = 1.0

            feature_vector = np.zeros(
                shape=(1, self.embedding_input_length, self.w2v_model.size),
                dtype=np.float
            )

            for i in range(self.embedding_input_length):
                feature_vector[0, i] = self.w2v_model.get_vector(feature[i])

            idx += 1
            feature_samples.append(feature_vector)
            label_samples.append(label_vector)

        return feature_samples, label_samples

    def _train_valid_test_split(self) -> None:
        """
        To split training,validation and testing data.
        :return: None
        """
        if 1 <= self._valid_size + self._test_size or self._data_size <= 2:
            log.error('parameter error:{0}+{1}>=1 or data size:{2} <= 2'
                      .format(self._valid_size, self._test_size, self._data_size))
            return
        train_valid_x, test_x = train_test_split(self.poetrys_vector, test_size=self._test_size, random_state=0)
        self.poetrys_vector_test = test_x
        self.data_size_test = len(test_x)
        self._data_index_test = np.arange(self.data_size_test)

        train_x, valid_x = train_test_split(train_valid_x, test_size=self._valid_size, random_state=0)
        self.poetrys_vector_train = train_x
        self.data_size_train = len(train_x)
        self._data_index_train = np.arange(self.data_size_train)

        self.poetrys_vector_valid = valid_x
        self.data_size_valid = len(valid_x)
        self._data_index_valid = np.arange(self.data_size_valid)

    def _data_batch(self, start, end, mode='one-hot', batch_type='train'):
        """
        To generate samples using batch.
        :param start: start position.
        :param end: end position.
        :param mode: encoding mode, one-hot or word2vec.
        :param batch_type: batch type,train,valid or test.
        :return: feature and label batches.
        """
        feature_batches = []
        label_batches = []
        sample = []
        feature_samples = None
        label_samples = None

        for i in range(start, end):
            if TRAIN_SET == batch_type:
                sample = self.poetrys_vector_train[self._data_index_train[i]]
            elif VALID_SET == batch_type:
                sample = self.poetrys_vector_valid[self._data_index_valid[i]]
            elif TEST_SET == batch_type:
                sample = self.poetrys_vector_test[self._data_index_test[i]]

            if ONE_HOT == mode:
                feature_samples, label_samples = self._one_hot_encoding(sample)
            elif WORD2VEC == mode:
                feature_samples, label_samples = self._word2vec_encoding(sample)
            feature_batches = feature_batches + feature_samples
            label_batches = label_batches + label_samples

        return feature_batches, label_batches

    def next_batch(self, batch_size=32, mode='one-hot', batch_type='train'):
        """
        To generate batch with yield.
        :param batch_size: size of batch.
        :param mode: encoding mode, one-hot or word2vec.
        :param batch_type: batch type,train,valid or test.
        :return: None
        """
        if TRAIN_SET == batch_type:
            n_chunk = self.data_size_train // batch_size
            log.debug('chunks:{0} training data size:{1}'.format(n_chunk, self.data_size_train))
        elif VALID_SET == batch_type:
            n_chunk = self.data_size_valid // batch_size
            log.debug('chunks:{0} validing data size:{1}'.format(n_chunk, self.data_size_valid))
        elif TEST_SET == batch_type:
            n_chunk = self.data_size_test // batch_size
            log.debug('chunks:{0} testing data size:{1}'.format(n_chunk, self.data_size_test))

        chunk_idx = 0
        while chunk_idx < n_chunk:
            start = chunk_idx * batch_size
            end = (chunk_idx + 1) * batch_size
            if TRAIN_SET == batch_type:
                np.random.shuffle(self._data_index_train)
                if end >= self.data_size_train:
                    end = self.data_size_train
            elif VALID_SET == batch_type:
                np.random.shuffle(self._data_index_valid)
                if end >= self.data_size_valid:
                    end = self.data_size_valid
            elif TEST_SET == batch_type:
                np.random.shuffle(self._data_index_test)
                if end >= self.data_size_test:
                    end = self.data_size_test

            feature_batches, label_batches = self._data_batch(start, end, mode, batch_type)

            if 0 == len(feature_batches):
                log.error("feature and label data are invalid. %d,%d", start, end)
                continue

            for i in range(0, len(feature_batches)):
                yield feature_batches[i], label_batches[i]

            chunk_idx += 1

    def load_word2vec_model(self, model_path) -> None:
        """
        To load word2vec model.
        :param model_path: Tthe path of model.
        :return: None.
        """
        if not os.path.exists(model_path):
            log.error("the path of word2vec model is not exist.")

        self.w2v_model = Word2vecModel()
        if not self.w2v_model.load(model_path):
            log.error("loading word2vec model error.")

    def idxlst2sentence(self, sample):
        """
        To convert index list to word string.
        :param sample: index list, such as "[321, 4721, 400, 3814, 282, 4999]".
        :return: word string, such as "共题诗句遍，".
        """
        return ''.join(reduce(operator.add, [list(self.i2w(poetry)) for poetry in sample]))

    def sentence2idxlist(self, sample):
        """
        To convert a word list to index list.
        :param sample: a word list such as "争空谁上尽，".
        :return: a index list such as "[75, 3949, 5395, 14, 1298, 6833]".
        """
        return reduce(operator.add, [list(map(self.w2i, poetry)) for poetry in sample])

    def sentence2vec(self, sample, isword2idx=True, mode='one-hot'):
        """
        To convert a sentence to vector.
        :param sample: a sentence.
        :param isword2idx: isword2idx==False, the sample is already be a index list
                                                otherwise it need be converted to an index list.
        :param mode: one-hot or word2vec encoding.
        :return: encoded vector.
        """

        if isword2idx:
            sample_vector = self.sentence2idxlist(sample)
        else:
            sample_vector = sample

        if self.embedding_input_length != len(sample_vector):
            log.error("type or length of sample is invalid.")
            return None

        if ONE_HOT == mode:
            feature_vector = np.zeros(
                shape=(1, self.embedding_input_length, self.vocab_size),
                dtype=np.float
            )

            for i, f in enumerate(sample_vector):
                feature_vector[0, i, f] = 1.0

            return feature_vector

        elif WORD2VEC == mode:
            feature_vector = np.zeros(
                shape=(1, self.embedding_input_length, self.w2v_model.size),
                dtype=np.float
            )

            for i in range(self.embedding_input_length):
                feature_vector[0, i] = self.w2v_model.get_vector(sample_vector[i])

            return feature_vector

        log.error("mode must be word2vec or one-hot.{0}".format(mode))
        return None

    def dump_data(self) -> None:
        """
        To dump: poetry's words list, poetry's words vectors, poetry's words vectors for training,
                 poetry's words vectors for testing, poetry's words vectors for validation,
                 poetry's words vocabulary, poetry's word to index mapping,poetry's index to word mapping.
        :return: None
        """
        org_filename = self.dump_dir + 'poetrys_words.dat'
        self._dump_list(org_filename, self.poetrys)

        vec_filename = self.dump_dir + 'poetrys_words_vector.dat'
        self._dump_list(vec_filename, self.poetrys_vector)

        train_vec_filename = self.dump_dir + 'poetrys_words_train_vector.dat'
        self._dump_list(train_vec_filename, self.poetrys_vector_train)

        valid_vec_filename = self.dump_dir + 'poetrys_words_valid_vector.dat'
        self._dump_list(valid_vec_filename, self.poetrys_vector_valid)

        test_vec_filename = self.dump_dir + 'poetrys_words_test_vector.dat'
        self._dump_list(test_vec_filename, self.poetrys_vector_test)

        vocab_filename = self.dump_dir + 'poetrys_vocab.dat'
        self._dump_list(vocab_filename, list(self.vocab))

        w2i_filename = self.dump_dir + 'poetrys_word2index.dat'
        self._dump_dict(w2i_filename, self.word2idx)

        i2w_filename = self.dump_dir + 'poetrys_index2word.dat'
        self._dump_dict(i2w_filename, self.idx2word)


#
# m = PoetrysDataSet('/home/zhanglei/Gitlab/LstmApp/config/cfg.ini')
# m._one_hot_encoding([980, 4588, 2959, 1257, 506, 4999, 1743, 4278, 4893, 787, 1203, 2, 358, 4721, 4730, 1141, 1892, 4999, 1759, 4619, 4515, 3496, 1937, 2, 2864, 1865, 4651, 1719, 2987, 4999, 3209, 2166, 3301, 1695, 3510, 2, 3487, 2673, 358, 4604, 493, 4999, 3196, 1906, 1112, 3588, 1845, 2])
# m.sentence2vec(sample="钟鼓寒，楼阁",isword2idx=True)
