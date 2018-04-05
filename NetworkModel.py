from keras import backend as K
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

from CustomZeroPadding import CustomZeroPadding2D

K.set_image_dim_ordering('th')


def build_model(input_shape, max_length, alphabet_length):
    model = Sequential()

    # 1 layer
    model.add(Convolution2D(64, (5, 5), input_shape=input_shape, activation='relu', padding='same',
                            kernel_initializer='glorot_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 2 layer
    model.add(Convolution2D(128, (5, 5), padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 3 layer
    model.add(Convolution2D(256, (3, 3),
                            padding='same', activation='relu', kernel_initializer='glorot_uniform'))

    # 3.5 layer - border_mode = 'same' preserves dimensionality
    model.add(Convolution2D(512, (3, 3),
                            padding='same', activation='relu', kernel_initializer='glorot_uniform'))

    # Need to zero pad one column on the right hand size of the output so pooling works
    # model.add(CustomZeroPadding2D(padding=(0, 0, 0, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Fourth layer - border_mode = 'same' preserves dimensionality
    model.add(Convolution2D(512, (3, 3),
                            padding='same', activation='relu', kernel_initializer='glorot_uniform'))

    # First Dense layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.5))

    # Second Dense layer
    model.add(Dense(4096, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.5))

    model.add(Dense(max_length * alphabet_length, activation='sigmoid'))
    return model
