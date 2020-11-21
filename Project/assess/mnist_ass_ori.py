from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
import tensorflow as tf

# 数据集
(X_train, y_train), (X_test, y_test) = mnist.load_data()  # 读取并划分MNIST训练集、测试集
'''
#用于1、2个模型
X_train = X_train.reshape(len(X_train), -1)  # 二维变一维
X_test = X_test.reshape(len(X_test), -1)
'''
#用于3~8个模型
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

X_train = X_train.astype('float32')  # 转为float类型
X_test = X_test.astype('float32')

X_train = (X_train - 127) / 127  # 灰度像素数据归一化
X_test = (X_test - 127) / 127

y_train = np_utils.to_categorical(y_train, num_classes=10)  # 独热编码。
y_test = np_utils.to_categorical(y_test, num_classes=10)
'''
# 定义模型
model_1 = tf.keras.models.load_model('MNIST/dnn_with_dropout.hdf5')  # 模型1
# 评估
loss, accuracy = model_1.evaluate(X_train[:10000], y_train[:10000])
print("dnn_with_dropout")
print('Test loss:', loss)
print('Accuracy:', accuracy)

model_2 = tf.keras.models.load_model('MNIST/dnn_without_dropout.hdf5')  # 模型2
loss, accuracy = model_2.evaluate(X_train[:10000], y_train[:10000])
print("dnn_without_dropout")
print('Test loss:', loss)
print('Accuracy:', accuracy)
'''
model_3 = tf.keras.models.load_model('MNIST/lenet5_with_dropout.hdf5')  # 模型3
loss, accuracy = model_3.evaluate(X_train[:10000], y_train[:10000])
print("lenet5_with_dropout")
print('Test loss:', loss)
print('Accuracy:', accuracy)

model_4 = tf.keras.models.load_model('MNIST/lenet5_without_dropout.hdf5')  # 模型4
loss, accuracy = model_4.evaluate(X_train[:10000], y_train[:10000])
print("lenet5_without_dropout")
print('Test loss:', loss)
print('Accuracy:', accuracy)

model_5 = tf.keras.models.load_model('MNIST/random1_mnist.h5')  # 模型5
loss, accuracy = model_5.evaluate(X_train[:10000], y_train[:10000])
print("random1_mnist")
print('Test loss:', loss)
print('Accuracy:', accuracy)

model_6 = tf.keras.models.load_model('MNIST/random2_mnist.h5')  # 模型6
loss, accuracy = model_6.evaluate(X_train[:10000], y_train[:10000])
print("random2_mnist")
print('Test loss:', loss)
print('Accuracy:', accuracy)

model_7 = tf.keras.models.load_model('MNIST/vgg16_with_dropout.hdf5')  # 模型7
loss, accuracy = model_7.evaluate(X_train[:10000], y_train[:10000])
print("vgg16_with_dropout")
print('Test loss:', loss)
print('Accuracy:', accuracy)

model_8 = tf.keras.models.load_model('MNIST/vgg16_without_dropout.hdf5')  # 模型8
loss, accuracy = model_8.evaluate(X_train[:10000], y_train[:10000])
print("vgg16_without_dropout")
print('Test loss:', loss)
print('Accuracy:', accuracy)