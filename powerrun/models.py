import os
import random
import numpy as np

from tensorflow.keras.layers import Dense, Input, Flatten, Conv1D, ReLU, ELU, LeakyReLU, MaxPool1D, AveragePooling1D, Dropout, \
    Conv2D, LeakyReLU, Reshape, BatchNormalization, Activation, Conv2DTranspose, Conv1DTranspose, Conv1D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, MaxPooling1D, concatenate, GlobalAveragePooling2D, \
    GlobalAveragePooling1D, GlobalMaxPooling1D
import tensorflow as tf

__version__ = 0.0012

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def get_resnet50v2_classification_model(input_shape=(360, 15, 1),
                                        filters=64,
                                        num_classes=2,
                                        ):

    input_shape = (*input_shape, 1)
    new_in = Input(shape=input_shape)
    heading = Reshape((int(input_shape[0]/4), input_shape[1]*4, 1,))(new_in)
    base_model = tf.keras.applications.ResNet50V2(include_top=False,
                                                  weights=None,
                                                  input_tensor=None,
                                                  input_shape=(int(input_shape[0]/4), input_shape[1]*4, 1,),
                                                  # input_shape=input_shape,
                                                  pooling=None,
                                                  classes=num_classes,
                                                  # **kwargs,
                                                  )
    base_model.layers.pop(0)
    base_model.trainable = True
    base_output = base_model(heading)
    # base_output = base_model(new_in)
    x = GlobalAveragePooling2D()(base_output)
    x = Dense(filters*8, activation='relu')(x)
    x = Dropout(0.35)(x)
    x = Dense(filters*4, activation='relu')(x)
    x = Dropout(0.35)(x)
    x = Dense(int(filters/2), activation='relu')(x)
    x_out = Dense(num_classes, activation='softmax')(x)
    new_model = tf.keras.models.Model(inputs=new_in, outputs=x_out)
    return new_model


def get_resnet1d_model(
                       input_shape=(40, 16,),
                       kernels=32,
                       stride=4
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


def get_resnet1d_model_tahn(
                            input_shape=(40, 16,),
                            kernels=64,
                            stride=4,
                            ):

    def residual_block_tahn(x, kernels, stride):
        out = Conv1D(kernels, stride, padding='same')(x)
        out = tf.keras.activations.tanh(out)
        out = Conv1D(kernels, stride, padding='same')(out)
        out = tf.keras.layers.add([x, out])
        out = tf.keras.activations.tanh(out)
        out = AveragePooling1D(2, 2)(out)
        return out

    def residual_block_avr(x, kernels, stride, pool=2):
        out = Conv1D(kernels, stride, padding='same')(x)
        out = ReLU()(out)
        out = Conv1D(kernels, stride, padding='same')(out)
        out = tf.keras.layers.add([x, out])
        out = ReLU()(out)
        out = AveragePooling1D(pool, 2)(out)
        # out = MaxPool1D(4, 2)(out)
        return out

    def residual_block_max(x, kernels, stride, pool=2):
        out = Conv1D(kernels, stride, padding='same')(x)
        out = ReLU()(out)
        out = Conv1D(kernels, stride, padding='same')(out)
        out = tf.keras.layers.add([x, out])
        out = ReLU()(out)
        out = MaxPool1D(pool, 2)(out)
        return out

    x_in = Input(shape=input_shape)
    x = Dense(32, activation="elu")(x_in)
    x = Conv1D(kernels, stride)(x)
    x1_1 = residual_block_avr(x, kernels, stride, pool=6)
    x1_2 = residual_block_avr(x1_1, kernels, stride, pool=5)
    x1_3 = residual_block_avr(x1_2, kernels, stride, pool=4)
    x1_4 = residual_block_avr(x1_3, kernels, stride, pool=3)
    x1_5 = residual_block_avr(x1_4, kernels, stride, pool=2)

    # x2 = residual_block_tahn(x, kernels, stride)
    # x2 = residual_block_tahn(x2, kernels, stride)
    # x2 = residual_block_tahn(x2, kernels, stride)
    # x2 = residual_block_tahn(x2, kernels, stride)

    x = concatenate([x, x1_1, x1_2, x1_3, x1_4, x1_5], axis=-2)
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.35)(x)
    x = Dense(32, activation="relu")(x)
    x_out = Dense(2, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=x_in, outputs=x_out)
    return model


def get_resnet1d_model_new(
                           input_shape=(360, 15),
                           filters=64,
                           stride=4,
                           num_classes=2,
                           ):

    def residual_block_max(x, filters, stride, pool=2):
        out = Conv1D(filters, stride, padding='same')(x)
        out = ReLU()(out)
        out = Conv1D(filters, stride, padding='same')(out)
        out = tf.keras.layers.add([x, out])
        out = ReLU()(out)
        out = MaxPool1D(pool, 2)(out)
        return out

    x_in = Input(shape=input_shape)
    x = Dense(filters, activation="relu")(x_in)
    x = Conv1D(filters, stride)(x)
    x1_1 = residual_block_max(x, filters, stride, pool=15)
    x1_2 = residual_block_max(x1_1, filters, stride, pool=10)
    x1_3 = residual_block_max(x1_2, filters, stride, pool=7)
    x1_4 = residual_block_max(x1_3, filters, stride, pool=5)
    x1_5 = residual_block_max(x1_4, filters, stride, pool=3)
    x1_6 = residual_block_max(x1_5, filters, stride, pool=2)

    x = concatenate([x, x1_1, x1_2, x1_3, x1_4, x1_5, x1_6], axis=-2)
    x = Flatten()(x)
    x = Dense(filters*4, activation="relu")(x)
    x = Dropout(0.35)(x)
    x = Dense(filters*2, activation="relu")(x)
    x = Dropout(0.35)(x)
    x = Dense(32, activation="relu")(x)
    x_out = Dense(num_classes, activation='softmax')(x)
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


def get_resnet1d_regression(
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
    x = Dense(32, activation="elu")(x)
    x = Dropout(0.35)(x)
    x = Dense(32, activation="elu")(x)
    x_out = Dense(1, activation='linear')(x)
    model = tf.keras.models.Model(inputs=x_in, outputs=x_out)
    return model


def get_resnet1d_and_regression_model(
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

    def residual_block_tahn(x, kernels, stride):
        out = Conv1D(kernels, stride, padding='same')(x)
        out = tf.keras.activations.tanh(out)
        out = Conv1D(kernels, stride, padding='same')(out)
        out = tf.keras.layers.add([x, out])
        out = tf.keras.activations.tanh(out)
        out = MaxPool1D(3, 2)(out)
        return out

    x_in = Input(shape=input_shape)

    x = Dense(units=24)(x_in)
    x = LeakyReLU()(x)
    x = Conv1D(kernels, stride)(x)
    x = residual_block_tahn(x, kernels, stride)
    x = residual_block_tahn(x, kernels, stride)
    x = residual_block_tahn(x, kernels, stride)
    x_base = residual_block_tahn(x, kernels, stride)

    x1 = Flatten()(x_base)
    x1 = Dense(units=32, activation=tf.keras.activations.tanh)(x1)
    x1 = Dropout(0.35)(x1)
    x1 = Dense(units=32, activation=tf.keras.activations.tanh)(x1)
    x_out1 = Dense(2, activation='softmax', name="trnd_dir")(x1)

    x2 = Dense(units=16)(x_in)
    x2 = LeakyReLU()(x2)
    x2 = Conv1D(kernels, stride)(x2)
    x2 = residual_block_tahn(x2, kernels, stride)
    x2 = residual_block_tahn(x2, kernels, stride)
    x2 = residual_block_tahn(x2, kernels, stride)
    x2_base = residual_block(x2, kernels, stride)

    x2 = Dense(units=32)(x2_base)
    x2 = ELU()(x2)
    x2 = Dropout(0.35)(x2)
    x2 = Dense(32, activation="linear")(x2)
    x_out2 = Dense(1, activation='linear', name='tcks_2chng')(x2)

    keras_model = tf.keras.models.Model(inputs=x_in, outputs=[x_out1, x_out2])
    return keras_model


