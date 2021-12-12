from tensorflow.keras.layers import Dense, Input, Flatten, Conv1D, ReLU, ELU, MaxPool1D, Dropout
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