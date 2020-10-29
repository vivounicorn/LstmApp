from src.lstm_model import LstmModel
from src.data_processing import PoetrysDataSet

cfg_file_path = '/home/zhanglei/Gitlab/LstmApp/config/cfg.ini'
base_data = PoetrysDataSet(cfg_file_path)
model = LstmModel(cfg_file_path, base_data)
# model.load('/home/zhanglei/Gitlab/LstmApp/data/models/model-479.hdf5')
# print(base_data.idxlst2sentence([321, 4721, 400, 3814, 282, 4999]))
# print(model.predict_base([321, 4721, 400, 3814, 282, 4999], False))
# print(model.predict_random())

base_data.dump_data()