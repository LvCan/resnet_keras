import keras
from data_generator import get_train_data,get_test_data
from resnet_model import resnet_101
import matplotlib.pyplot as plt
import time
def train(retrain=False,size=64,eps=10):
    if retrain:
        model=keras.models.load_model('model/train_model_101.h5')
        his = model.fit_generator(get_train_data(batch_size=size), steps_per_epoch=int(3295 / size), epochs=eps,
                                  validation_data=get_test_data(batch_size=size), validation_steps=int(1885 / size)) #9885
        model.save('model/train_model_101.h5')
    else:
        model=resnet_101()
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
        his=model.fit_generator(get_train_data(batch_size=size),steps_per_epoch=int(3295/size),epochs=eps,validation_data=get_test_data(batch_size=size),validation_steps=int(1885/size))
        model.save('model/train_model_101.h5')
    pred=model.evaluate_generator(get_test_data(batch_size=size),steps=int(9885/size))
    print(pred)
    loss=his.history['loss']
    acc=his.history['acc']
    val_acc=his.history['val_acc']
    plt.plot(loss,color='r',label='loss')
    plt.plot(acc,color='b',label='acc')
    plt.plot(val_acc,color='g',label='val_acc')
    plt.legend('best')
    plt.show()

if __name__=='__main__':
    start=time.time()

    train(retrain=True,size=64,eps=10)

    end=time.time()
    print('use time:')
    print(end-start)