from keras.datasets import cifar100
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
import tensorflow as tf

# 数据集
(X_train, y_train), (X_test, y_test) = cifar100.load_data()  # 读取并划分训练集、测试集

X_train = X_train.astype('float32')  # 转为float类型
X_test = X_test.astype('float32')

X_train=X_train/255.0
X_test=X_test/255.0#归一化


y_train = np_utils.to_categorical(y_train)  # 独热编码
y_test = np_utils.to_categorical(y_test)

# 定义模型
model_1 = tf.keras.models.load_model('cifar100/CNN_with_dropout.h5')  # 模型1
# 评估
loss, accuracy = model_1.evaluate(X_train[:10000], y_train[:10000])
print("CNN_with_dropout")
print('Test loss:', loss)
print('Accuracy:', accuracy)

model_2 = tf.keras.models.load_model('cifar100/CNN_without_dropout.h5')  # 模型2
loss, accuracy = model_2.evaluate(X_train[:10000], y_train[:10000])
print("CNN_without_dropout")
print('Test loss:', loss)
print('Accuracy:', accuracy)

model_3 = tf.keras.models.load_model('cifar100/lenet5_with_dropout.h5')  # 模型3
loss, accuracy = model_3.evaluate(X_train[:10000], y_train[:10000])
print("lenet5_with_dropout")
print('Test loss:', loss)
print('Accuracy:', accuracy)

model_4 = tf.keras.models.load_model('cifar100/lenet5_without_dropout.h5')  # 模型4
loss, accuracy = model_4.evaluate(X_train[:10000], y_train[:10000])
print("lenet5_without_dropout")
print('Test loss:', loss)
print('Accuracy:', accuracy)

model_5 = tf.keras.models.load_model('cifar100/random1.h5')  # 模型5
loss, accuracy = model_5.evaluate(X_train[:10000], y_train[:10000])
print("random1")
print('Test loss:', loss)
print('Accuracy:', accuracy)

model_6 = tf.keras.models.load_model('cifar100/random2.h5')  # 模型6
loss, accuracy = model_6.evaluate(X_train[:10000], y_train[:10000])
print("random2")
print('Test loss:', loss)
print('Accuracy:', accuracy)

model_7 = tf.keras.models.load_model('cifar100/ResNet_v1.h5')  # 模型7
loss, accuracy = model_7.evaluate(X_train[:10000], y_train[:10000])
print("ResNet_v1")
print('Test loss:', loss)
print('Accuracy:', accuracy)

model_8 = tf.keras.models.load_model('cifar100/ResNet_v2.h5')  # 模型8
loss, accuracy = model_8.evaluate(X_train[:10000], y_train[:10000])
print("ResNet_v2")
print('Test loss:', loss)
print('Accuracy:', accuracy)