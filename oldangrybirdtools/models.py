from tensorflow.keras.layers import Dense, Input, Flatten, Conv1D, ReLU, ELU, MaxPool1D, Dropout, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, concatenate
from tensorflow.keras.models import Model


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

    input_shape = (*input_shape, 1)
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

    return Model(input_layer, x_out)


def get_old_model(input_shape):
    def base_layer(input, filters, k_size, pool_size=(2, 1), act='elu'):
        x = Conv2D(filters, k_size, padding='same', activation=act)(input)
        xa = AveragePooling2D(pool_size=pool_size, padding='same')(x)
        xm = MaxPooling2D(pool_size=pool_size, padding='same')(x)
        xa = Conv2D(filters, k_size, padding='same', activation=act)(xa)
        xm = Conv2D(filters, k_size, padding='same', activation=act)(xm)
        x = concatenate([xa, xm])
        x = Conv2D(filters, k_size, padding='same', activation=act)(x)
        return x

    k_size = (3, 3)

    input_shape = (*input_shape, 1)
    input_layer = Input(shape=input_shape)

    x = Conv2D(64, k_size, padding='same', activation='elu')(input_layer)
    x = base_layer(x, 64, k_size, act='elu')
    x = base_layer(x, 128, k_size, act='elu')
    x = base_layer(x, 64, k_size, act='elu')
    x = Conv2D(32, k_size, padding='same', activation='elu')(x)
    x = Conv2D(8, k_size, padding='same', activation='elu')(x)
    x = Flatten()(x)
    x = Dense(12, activation='tanh')(x)
    x = Dense(2, activation='softmax')(x)

    return Model(input_layer, x)