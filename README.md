# automation_testing
automation_testing
自动化测试大作业

选题方向：AI

------

算法释义，所用第三方库及版本详见Report
程序入口
程序结构
Project
|-- assess测试数据质量评估
|   |-- MNIST(8个模型，太大没有上传)
|   |-- cifar100(同上)
|   |-- cifar_ass_change.py 对生成的cifar_100测试数据的评估
|   |-- cifar_ass_ori.py cifar_100原始数据集评估
|   |-- mnist_ass_change.py 对生成的mnist测试数据的评估
|   |-- mnist_ass_ori.py mnist原始数据集评估
|-- test_data测试数据生成
    |-- cifar100_main.py cifar100 生成测试数据集的方法
    |-- cifar100_transform.py 将数据集转化为图片
    |-- mnist_main.py mnist 生成测试数据集的方法
    |-- mnist_transform.py 将数据集转化为图片

