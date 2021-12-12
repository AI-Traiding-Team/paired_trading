from tensorflow.keras.layers import Dense, Input, Flatten, Conv1D, ReLU, ELU, MaxPool1D, Dropout, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, concatenate
import tensorflow as tf

__version__ = 0.0001


def get_resnet1d_model(
                       input_shape=(40, 16,),
                       kernels=32,
                       stride=4,
                       ):
    def residual_block(x, kernels, stride):
        out = Conv1D(kernels, stride, padding='same')(x)
        out = ReLU()(out)
        out = Conv1D(kernels, stride, padding='same')(out)
        out = tf.keras.layers.add([x, out])
        out = ReLU()(out)
        out = MaxPool1D(3, 2)(out)
        return out

    x_in = Input(shape=input_shape)
    x = Dense(16, activation="elu")(x_in)
    x = Conv1D(kernels, stride)(x)
    x = residual_block(x, kernels, stride)
    x = residual_block(x, kernels, stride)
    x = residual_block(x, kernels, stride)
    x = residual_block(x, kernels, stride)
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.35)(x)
    x = Dense(32, activation="relu")(x)
    x_out = Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=x_in, outputs=x_out)
    return model


def get_angry_bird_model(input_shape):
    def conv_layer(input, n, k_size=(3, 5), separate=False):
        layer = Conv2D(n, k_size, padding='same', activation='elu')
        if separate:
            output = layer(input), layer(input)
        else:
            output = layer(layer(input))
        return output

    def pooling_layer(input, pool_size=(2, 1)):
        avg_l = AveragePooling2D(pool_size=pool_size, padding='same')
        max_l = MaxPooling2D(pool_size=pool_size, padding='same')
        return avg_l(input), max_l(input)

    input_layer = Input(shape=input_shape)
    x = conv_layer(input_layer, n=64)
    xa, xb = pooling_layer(x)
    xa, _ = conv_layer(xa, n=64, separate=True)
    xb, _ = conv_layer(xb, n=64, separate=True)
    x = concatenate([xa, xb])
    x = conv_layer(x, n=128)
    xa, xb = pooling_layer(x)
    xa, xb = conv_layer(x, n=64, separate=True)
    x = concatenate([xa, xb])
    x = conv_layer(x, n=128)
    xa, xb = pooling_layer(x)
    xa, _ = conv_layer(xa, n=64, separate=True)
    xb, _ = conv_layer(xb, n=64, separate=True)
    x = concatenate([xa, xb])
    x, _ = conv_layer(x, n=32, separate=True)
    x, _ = conv_layer(x, n=8, separate=True)
    x = Flatten()(x)
    x = Dense(12, activation='tanh')(x)
    x_out = Dense(2, activation='softmax')(x)

    return tf.keras.models.Model(input_layer, x_out)
