import Augmentor
p = Augmentor.Pipeline("../../Data/mnist/test")


#增强操作
# 1.旋转 向左最大旋转角度10，向右最大旋转角度10
#p.rotate(probability=1,max_left_rotation=10, max_right_rotation=10)

#2.透视、形变（上下左右方向的垂直形变）
#p.skew_tilt(probability=1,magnitude=0.4)

#3.弹性扭曲
#p.random_distortion(probability=1, grid_height=3, grid_width=3, magnitude=6)

# 4放大 最小为1.1倍，最大为1.6倍
#p.zoom(probability=1, min_factor=1.1, max_factor=1.6)

#5 随机擦除   随机擦除是一种使模型对遮挡更加鲁棒的技术。这个对使用神经网络训练物体检测的时候非常有用：
#p.random_erasing(probability=1,rectangle_area=0.2)



#6.组合

p.rotate(probability=1,max_left_rotation=10, max_right_rotation=10)
p.skew_tilt(probability=1,magnitude=0.4)
p.random_distortion(probability=1, grid_height=3, grid_width=3, magnitude=6)
p.zoom(probability=1, min_factor=1.1, max_factor=1.6)
p.random_erasing(probability=1,rectangle_area=0.2)




p.process()#每个输出一个


'''
# 指定增强后图片数目总量
p.sample(10000)
'''