import Augmentor
p = Augmentor.Pipeline("../../Data/cifar_100/test")


# 增强操作
# 1放大 最小为1.1倍，最大为1.5倍
#p.zoom(probability=1, min_factor=1.1, max_factor=1.5)

#2.透视、形变（上下左右方向的垂直形变） 鲁棒性
#p.skew_tilt(probability=1,magnitude=0.4)

#3.弹性扭曲
#p.random_distortion(probability=1, grid_height=3, grid_width=3, magnitude=6)

#4 随机擦除   随机擦除是一种使模型对遮挡更加鲁棒的技术。这个对使用神经网络训练物体检测的时候非常有用：
#p.random_erasing(probability=1,rectangle_area=0.3)

#5 改变颜色
#p.random_color(probability=1,min_factor=1.5,max_factor=6)

#6 改变亮度
p.random_brightness(probability=1,min_factor=1.1,max_factor=1.8)


p.process()#每个输出一个