# resnet_keras
# 使用方式：
在resnet_model中修改所使用的模型的input_shape和classes数目，train_50/train_101/train_152分别对应resnet_50/resnet_101/resnet152的训练。根据自己的数据集修改data_generator，本实验所用的数据集保存方式为每行格式如下的txt：

image_name   label

也可根据自己数据集进行修改
