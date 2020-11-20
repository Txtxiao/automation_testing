import cv2
import numpy as np
import os
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
import tensorflow as tf
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 数据集
dir="E:/大三上/自动化/automation_testing/Data/mnist/zoom/"
name=os.listdir(dir)

name.sort(key=lambda x:int(x[14:-45]))
imgs = []



def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    ##cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img

for i in name:
     imgs.append(cv_imread(dir + i))
imgs = np.array(imgs)


#imgs = imgs.reshape(len(imgs), -1)  # 二维变一维,用于1、2个模型

imgs= imgs.reshape(-1, 28, 28, 1)#用于3~8个模型

#print(imgs)
imgs = imgs.astype('float32')  # 转为float类型

imgs = (imgs - 127) / 127  # 灰度像素数据归一化
#print(imgs)

y_train = np_utils.to_categorical(y_train)  # 独热编码

'''
# 定义模型
model_1 = tf.keras.models.load_model('mnist/dnn_with_dropout.hdf5')  # 模型1
# 评估
loss, accuracy = model_1.evaluate(imgs, y_train[:10000])
print("dnn_with_dropout")
print('Test loss:', loss)
print('Accuracy:', accuracy)

model_2 = tf.keras.models.load_model('MNIST/dnn_without_dropout.hdf5')  # 模型2
loss, accuracy = model_2.evaluate(imgs, y_train[:10000])
print("dnn_without_dropout")
print('Test loss:', loss)
print('Accuracy:', accuracy)
'''
model_3 = tf.keras.models.load_model('MNIST/lenet5_with_dropout.hdf5')  # 模型3
loss, accuracy = model_3.evaluate(imgs, y_train[:10000])
print("lenet5_with_dropout")
print('Test loss:', loss)
print('Accuracy:', accuracy)

model_4 = tf.keras.models.load_model('MNIST/lenet5_without_dropout.hdf5')  # 模型4
loss, accuracy = model_4.evaluate(imgs, y_train[:10000])
print("lenet5_without_dropout")
print('Test loss:', loss)
print('Accuracy:', accuracy)

model_5 = tf.keras.models.load_model('MNIST/random1_mnist.h5')  # 模型5
loss, accuracy = model_5.evaluate(imgs, y_train[:10000])
print("random1_mnist")
print('Test loss:', loss)
print('Accuracy:', accuracy)

model_6 = tf.keras.models.load_model('MNIST/random2_mnist.h5')  # 模型6
loss, accuracy = model_6.evaluate(imgs, y_train[:10000])
print("random2_mnist")
print('Test loss:', loss)
print('Accuracy:', accuracy)

model_7 = tf.keras.models.load_model('MNIST/vgg16_with_dropout.hdf5')  # 模型7
loss, accuracy = model_7.evaluate(imgs, y_train[:10000])
print("vgg16_with_dropout")
print('Test loss:', loss)
print('Accuracy:', accuracy)

model_8 = tf.keras.models.load_model('MNIST/vgg16_without_dropout.hdf5')  # 模型8
loss, accuracy = model_8.evaluate(imgs, y_train[:10000])
print("vgg16_without_dropout")
print('Test loss:', loss)
print('Accuracy:', accuracy)



