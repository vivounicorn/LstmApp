from src.lstm_model import LstmModel
from src.data_processing import PoetrysDataSet
from src.word2vec import Word2vecModel

cfg_file_path = '/home/zhanglei/Gitlab/LstmApp/config/cfg.ini'
base_data = PoetrysDataSet(cfg_file_path)
# model = LstmModel(cfg_file_path, base_data)
#
# model.load('/home/zhanglei/Gitlab/LstmApp/data/models/model-59.hdf5')
# print(base_data.idxlst2sentence([321, 4721, 400, 3814, 282, 4999]))
# print(model.predict_base([321, 4721, 400, 3814, 282, 4999], False))
# print(model.predict_random())
# s="争空谁上尽，"
# sv=base_data.sentence2idxlist(s)
# print(sv)
# generate = model.predict_sen(sv)
# print(generate)
# print(model.dataset.idxlst2sentence(generate))
base_data.dump_data()
# # VecTrining anxiang1836


w2v = Word2vecModel()
w2v.train_vec()
w2v.load('../data//w2v_models/text_type_1.model')
a=w2v.most_similar(str(base_data.w2i('床')))
for i in range(len(a)):
    print(base_data.i2w(a[i][0]),a[i][1])
#
# a=w2v.get_vector(str(base_data.w2i('床')))
# b=w2v.get_vector(str(base_data.w2i('上')))
# print(a)
# print(a.shape)
# import numpy as np
# feature_vector = np.zeros(
#                 shape=(1, 6, 50),
#                 dtype=np.float
#             )
#
# for i in range(6):
#     feature_vector[0, i] = w2v.get_vector(str(base_data.w2i('床')))
# feature_vector[0,0]=a
# feature_vector[0,1]=b
# print(feature_vector)