import cv2
import numpy as np
import os
import pylab
from keras.datasets import cifar100
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
import tensorflow as tf
(X_train, y_train), (X_test, y_test) = cifar100.load_data()

# 数据集

dir="../../Data/cifar_100/zoom/"
name=os.listdir(dir)
name.sort(key=lambda x:int(x[14:-45]))
imgs = []

for i in name:
     image=cv2.imread(dir + i,cv2.IMREAD_COLOR)
     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     imgs.append(image)


imgs = np.array(imgs)
#pylab.imshow(imgs[0])
#pylab.show()


imgs = imgs.astype('float32')  # 转为float类型
imgs=imgs/255.0

#print(imgs)

y_train = np_utils.to_categorical(y_train)  # 独热编码


# 定义模型
model_1 = tf.keras.models.load_model('cifar100/CNN_with_dropout.h5')  # 模型1
# 评估
loss, accuracy = model_1.evaluate(imgs, y_train[:10000])
print("CNN_with_dropout")
print('Test loss:', loss)
print('Accuracy:', accuracy)

model_2 = tf.keras.models.load_model('cifar100/CNN_without_dropout.h5')  # 模型2
loss, accuracy = model_2.evaluate(imgs, y_train[:10000])
print("CNN_without_dropout")
print('Test loss:', loss)
print('Accuracy:', accuracy)

model_3 = tf.keras.models.load_model('cifar100/lenet5_with_dropout.h5')  # 模型3
loss, accuracy = model_3.evaluate(imgs, y_train[:10000])
print("lenet5_with_dropout")
print('Test loss:', loss)
print('Accuracy:', accuracy)

model_4 = tf.keras.models.load_model('cifar100/lenet5_without_dropout.h5')  # 模型4
loss, accuracy = model_4.evaluate(imgs, y_train[:10000])
print("lenet5_without_dropout")
print('Test loss:', loss)
print('Accuracy:', accuracy)

model_5 = tf.keras.models.load_model('cifar100/random1.h5')  # 模型5
loss, accuracy = model_5.evaluate(imgs, y_train[:10000])
print("random1")
print('Test loss:', loss)
print('Accuracy:', accuracy)

model_6 = tf.keras.models.load_model('cifar100/random2.h5')  #模型6
loss, accuracy = model_6.evaluate(imgs, y_train[:10000])
print("random2")
print('Test loss:', loss)
print('Accuracy:', accuracy)

model_7 = tf.keras.models.load_model('cifar100/ResNet_v1.h5')  # 模型7
loss, accuracy = model_7.evaluate(imgs, y_train[:10000])
print("ResNet_v1")
print('Test loss:', loss)
print('Accuracy:', accuracy)

model_8 = tf.keras.models.load_model('cifar100/ResNet_v2.h5')  # 模型8
loss, accuracy = model_8.evaluate(imgs, y_train[:10000])
print("ResNet_v2")
print('Test loss:', loss)
print('Accuracy:', accuracy)
