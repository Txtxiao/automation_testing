import struct
from array import array
import os
import scipy.misc
# 通过 pip install pypng 命令安装此库
import png
from keras.datasets import cifar100
from matplotlib import pyplot
from scipy.misc import toimage
from PIL import Image

(X_train, y_train), (X_test, y_test) = cifar100.load_data()
#print(X_train.shape)
#print(X_train[0,:])

save_dir = '../../Data/cifar_100/test/'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)
#print(y_train[0,:])
index=0
for x, y in zip(X_train, y_train):
    label = y[0]

    img_path = save_dir + '%d.jpg' % index
    index += 1
    print(index)
    img = Image.fromarray(x)
    img.save(img_path)
    if(index==10000):
        break



