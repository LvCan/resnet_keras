from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
datagen=ImageDataGenerator(rescale=1/255,rotation_range=50,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)


with open(r'E:\competition\test\comp\train.txt') as file:
    train_data=file.readlines()
np.random.shuffle(train_data)
with open(r'E:\competition\test\comp\test.txt') as file:
    test_data=file.readlines()
np.random.seed(0)
np.random.shuffle(test_data)
def convert_to_one(classes,label):
    one=np.eye(classes)[label]
    return one
def get_train_data(classes=659,batch_size=32):
    x=0
    while True:
        fig_all=[]
        label_all=[]
        for i in range(batch_size):
            x %= len(train_data)
            fig=np.array(Image.open(train_data[x].split('   ')[0]))
            fig=fig.reshape(105,105,1)
            label=int(train_data[x].split('   ')[1])
            one=convert_to_one(classes,label)
            fig_all.append(fig)
            label_all.append(one)
            x+=1
        fig_all=np.array(fig_all)
        label_all=np.array(label_all)
        datagen.fit(fig_all)
        yield next(datagen.flow(fig_all,label_all,batch_size=batch_size))
def get_test_data(batch_size=32):
    x = 0
    while True:
        fig_all = []
        label_all = []
        for i in range(batch_size):
            x %= len(test_data)
            fig = np.array(Image.open(test_data[x].split('   ')[0]))
            fig = fig.reshape(105, 105, 1)
            label = int(test_data[x].split('   ')[1])
            one = convert_to_one(659, label)
            fig_all.append(fig)
            label_all.append(one)
            x += 1
        fig_all = np.array(fig_all)
        label_all = np.array(label_all)
        yield fig_all, label_all
if __name__=='__main__':
    for i in get_train_data():
        print(i)