#数据下载地址：http://download.tensorflow.org/example_images/flower_photos.tgz
#加载相关模块
from skimage import io,transform
from pandas import Series, DataFrame
import glob
import os
import numpy as np
from keras.models import Sequential
from keras.layers.core import Flatten,Dense,Dropout
from keras.layers.convolutional import Convolution2D,MaxPooling2D,ZeroPadding2D
from keras.optimizers import SGD,Adadelta,Adagrad
from keras.utils import np_utils,generic_utils
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.core import Dense, Dropout, Activation, Flatten
from six.moves import range

path='../data/flower_photos' #修改为自己的解压的图片路径
w=182
h=182
c=3


# 读取图片
def read_img(path):
    cate = [path + '/' + x for x in os.listdir(path) if os.path.isdir(path + '/' + x)]
    print(cate)
    imgs = []
    labels = []
    n = 0
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            # print('reading the images:%s'%(im))
            img = io.imread(im)
            # print('before resize:',img.shape)  #
            img = transform.resize(img, (w, h))
            # print('after:',img.shape)
            imgs.append(img)
            labels.append(idx)
            n = n + 1

    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


# 调用函数
data, label = read_img(path)
# print('叠加之后的形状：',data.shape)

#打乱顺序,将标签转为二进制独热形式（0和1组成）
num_example=data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
label=label[arr]
from keras.utils.np_utils import to_categorical
labels_5= to_categorical(label,num_classes=5)
print(labels_5)

from sklearn.model_selection import train_test_split
x_train, y_test, x_label, y_label = train_test_split(data,labels_5, test_size=0.3, random_state=42)

def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False, aug=None):
    while 1:  # 要无限循环
        assert len(inputs) == len(targets)#判断输入数据长度和label长度是否相同
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range( len(inputs) - batch_size ):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
                if aug is not None:
                  (inputs[excerpt], targets[excerpt]) = next(aug.flow(inputs[excerpt],targets[excerpt], batch_size=batch_size))
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
                if aug is not None:
                  (inputs[excerpt], targets[excerpt]) = next(aug.flow(inputs[excerpt],targets[excerpt], batch_size=batch_size))
            yield inputs[excerpt], targets[excerpt]#每次产生batchsize个数据

from keras.preprocessing.image import ImageDataGenerator
# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")

model = Sequential() #第一个卷积层，4个卷积核，每个卷积核大小5*5。
#激活函数用tanh #你还可以在model.add(Activation('tanh'))后加上dropout的技巧: model.add(Dropout(0.5))
model.add(Convolution2D(4, 5, 5,input_shape=(w, h,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #第二个卷积层，8个卷积核，每个卷积核大小3*3。
#激活函数用tanh #采用maxpooling，poolsize为(2,2)
model.add(Convolution2D(8, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#第三个卷积层，16个卷积核，每个卷积核大小3*3 #激活函数用tanh
#采用maxpooling，poolsize为(2,2)
model.add(Convolution2D(16, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#全连接层，先将前一层输出的二维特征图flatten为一维的。
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#多分类
model.add(Dense(5)) # 共有5个类别
model.add(Activation('softmax'))
#print(model.summary())
model.compile(loss='categorical_crossentropy',optimizer='adam')#使用分类交叉熵（categorical_crossentropy），因为我们有超过2个类别，否则将使用二进制交叉熵（binary crossentropy ）。

#model.fit(data,labels_5,epochs=6,batch_size=2,verbose=2)#旧方法不再适用
H=model.fit_generator(minibatches(x_train,x_label,batch_size=6,shuffle=False,aug=aug),
                            steps_per_epoch=len(x_train)//6,
                            validation_data=minibatches(y_test,y_label,batch_size=6,shuffle=False,aug=None),
                            validation_steps=len(y_test)//6,
                            epochs=6)
#model.train_on_batch(minibatches(data, labels_5, batch_size=6, shuffle=False))

import matplotlib.pyplot as plt
N = 6 # N=epochs
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("plot.png")