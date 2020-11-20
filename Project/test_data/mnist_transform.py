import struct
from array import array
import os
import scipy.misc
import png
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

save_dir = '../../Data/mnist/test/'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

# 保存前10000张图片
for i in range(10000):
    #表示第i张图片（序号从0开始）
    image_array = X_train[i, :]
    # 把MNIST还原为28x28维的图像。
    image_array = image_array.reshape(28, 28)
    # 保存文件的格式为 0.jpg, 1.jpg, ...
    filename = save_dir + '%d.jpg' % i
    # 将image_array保存为图片
    # 先用scipy.misc.toimage转换为图像，再调用save直接保存。
    scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)
    print(i)
