import keras
from data_generator import get_train_data,get_test_data
from resnet_model import resnet_152
import matplotlib.pyplot as plt
import time
def train(retrain=False):
    if retrain:
        model=keras.models.load_model('model/train_model.h5')
        his = model.fit_generator(get_train_data(), steps_per_epoch=int(3295 / 256), epochs=10,
                                  validation_data=get_test_data(), validation_steps=int(1885 / 256)) #9885
        model.save('model/train_model.h5')
    else:
        model=resnet_152()
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
        his=model.fit_generator(get_train_data(),steps_per_epoch=int(3295/32),epochs=10,validation_data=get_test_data(),validation_steps=int(1885/32))
        model.save('model/train_model.h5')
    pred=model.evaluate_generator(get_test_data(),steps=int(2885/32))
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

    train(retrain=True)

    end=time.time()
    print('use time:')
    print(end-start)