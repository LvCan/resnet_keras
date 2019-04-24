from keras.layers import Conv2D
from functools import wraps
import keras
from keras.layers import Activation,LeakyReLU,BatchNormalization
from keras.regularizers import l2

@wraps(Conv2D)
def my_conv(*args,**kwargs):
    new_kwargs={'kernel_regularizer':l2(5e-5)}
    new_kwargs['padding']='same'
    new_kwargs['kernel_initializer']=keras.initializers.glorot_uniform(seed=0)
    new_kwargs.update(kwargs)
    return Conv2D(*args,**new_kwargs)
def identity_block(x,filter):
    f1,f2,f3=filter
    x_short=x
    x=my_conv(filters=f1,kernel_size=(1,1),strides=(1,1))(x)
    x=keras.layers.BatchNormalization(axis=-1)(x)
    x=LeakyReLU(alpha=0.05)(x)

    x=my_conv(filters=f2,kernel_size=(3,3),strides=(1,1))(x)
    x=BatchNormalization(axis=-1)(x)
    x=LeakyReLU(alpha=0.05)(x)

    x=my_conv(filters=f3,kernel_size=(1,1),strides=(1,1))(x)
    x=BatchNormalization(axis=-1)(x)

    x=keras.layers.Add()([x,x_short])
    x=LeakyReLU(alpha=0.05)(x)
    return x
def convolutional_block(x,filter,stride=(2,2)):
    f1,f2,f3=filter
    x_short=x
    x=my_conv(filters=f1,kernel_size=(1,1),strides=stride)(x)
    x=BatchNormalization(axis=-1)(x)
    x=LeakyReLU(alpha=0.05)(x)

    x=my_conv(filters=f2,kernel_size=(3,3),strides=(1,1))(x)
    x=BatchNormalization(axis=-1)(x)
    x=LeakyReLU(alpha=0.05)(x)

    x=my_conv(filters=f3,kernel_size=(1,1),strides=(1,1))(x)
    x=BatchNormalization(axis=-1)(x)

    x_short=my_conv(filters=f3,kernel_size=(3,3),strides=stride)(x_short)
    x_short=BatchNormalization(axis=-1)(x_short)
    x=keras.layers.Add()([x,x_short])
    x=LeakyReLU(alpha=0.05)(x)
    return x
def resnet_152(input_shape=(105,105,1),classes=659):
    x_input=keras.Input(input_shape)
    x=my_conv(filters=64,kernel_size=(7,7),strides=(2,2))(x_input)
    x=BatchNormalization(axis=-1)(x)
    x=LeakyReLU(alpha=0.05)(x)
    x=keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)

    x=convolutional_block(x,(64,64,256),stride=(1,1))
    x=identity_block(x,(64,64,256))
    x = identity_block(x, (64, 64, 256))

    x=convolutional_block(x,(128,128,512))
    x=identity_block(x,(128,128,512))
    x = identity_block(x, (128, 128, 512))
    x = identity_block(x, (128, 128, 512))
    x = identity_block(x, (128, 128, 512))
    x = identity_block(x, (128, 128, 512))
    x = identity_block(x, (128, 128, 512))
    x = identity_block(x, (128, 128, 512))

    x=convolutional_block(x,(256,256,1024))
    for i in range(35):
        x=identity_block(x,(256,256,1024))

    x=convolutional_block(x,(512,512,2048))
    for i in range(2):
        x=identity_block(x,(512,512,2048))

    x=keras.layers.AveragePooling2D((4,4))(x)
    x=keras.layers.Flatten()(x)
    x=keras.layers.Dense(classes,activation='softmax',kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    model=keras.models.Model(inputs=x_input,outputs=x)
    return model
def resnet_101(input_shape=(105,105,1),classes=659):
    x_input=keras.Input(input_shape)
    x=my_conv(filters=64,kernel_size=(7,7),strides=(2,2))(x_input)
    x=BatchNormalization(axis=-1)(x)
    x=LeakyReLU(alpha=0.05)(x)
    x=keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)

    x=convolutional_block(x,(64,64,256),stride=(1,1))
    x=identity_block(x,(64,64,256))
    x = identity_block(x, (64, 64, 256))

    x=convolutional_block(x,(128,128,512))
    for i in range(3):
        x=identity_block(x,(128,128,512))

    x=convolutional_block(x,(256,256,1024))
    for i in range(22):
        x=identity_block(x,(256,256,1024))

    x=convolutional_block(x,(512,512,2048))
    for i in range(2):
        x=identity_block(x,(512,512,2048))

    x=keras.layers.AveragePooling2D((4,4))(x)
    x=keras.layers.Flatten()(x)
    x=keras.layers.Dense(classes,activation='softmax',kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    model=keras.models.Model(inputs=x_input,outputs=x)
    return model
def resnet_50(input_shape=(105,105,1),classes=659):
    x_input=keras.Input(input_shape)
    x=my_conv(filters=64,kernel_size=(7,7),strides=(2,2))(x_input)
    x=BatchNormalization(axis=-1)(x)
    x=LeakyReLU(alpha=0.05)(x)
    x=keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)

    x=convolutional_block(x,(64,64,256),stride=(1,1))
    x=identity_block(x,(64,64,256))
    x = identity_block(x, (64, 64, 256))

    x=convolutional_block(x,(128,128,512))
    for i in range(3):
        x=identity_block(x,(128,128,512))

    x=convolutional_block(x,(256,256,1024))
    for i in range(5):
        x=identity_block(x,(256,256,1024))

    x=convolutional_block(x,(512,512,2048))
    for i in range(2):
        x=identity_block(x,(512,512,2048))

    x=keras.layers.AveragePooling2D((4,4))(x)
    x=keras.layers.Flatten()(x)
    x=keras.layers.Dense(classes,activation='softmax',kernel_initializer=keras.initializers.glorot_uniform(seed=0))(x)
    model=keras.models.Model(inputs=x_input,outputs=x)
    return model