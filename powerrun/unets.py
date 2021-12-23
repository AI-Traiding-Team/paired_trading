from tensorflow.keras.layers import Dense, Input, Flatten, Conv1D, ReLU, ELU, LeakyReLU, MaxPool1D, AveragePooling1D, Dropout, \
    Conv2D, LeakyReLU, Reshape, BatchNormalization, Activation, Conv2DTranspose, Conv1DTranspose, Conv1D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, MaxPooling1D, concatenate, GlobalAveragePooling2D, \
    GlobalAveragePooling1D, GlobalMaxPooling1D

import tensorflow as tf

__version__ = 0.0005


def get_unet1d_base(
                    input_shape=(360, 15),
                    filters=64,
                    kernels=5,
                    num_classes=2,
                    ):

    x_in = Input(input_shape)  # Создаем входной слой с размерностью input_shape

    # Block 1
    x = Conv1D(filters, kernels, padding='same', name='block1_conv1')(x_in)  # Добавляем Conv2D-слой с 64-нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv1D(filters, kernels, padding='same', name='block1_conv2')(x)  # Добавляем Conv2D-слой с 64-нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_1_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_1_out

    x = MaxPooling1D()(block_1_out)  # Добавляем слой MaxPooling2D

    # Block 2
    x = Conv1D(filters*2, kernels, padding='same', name='block2_conv1')(x)  # Добавляем Conv2D-слой с 128-нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv1D(filters*2, kernels, padding='same', name='block2_conv2')(x)  # Добавляем Conv2D-слой с 128-нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_2_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_2_out

    x = MaxPooling1D()(block_2_out)  # Добавляем слой MaxPooling2D

    # Block 3
    x = Conv1D(filters*4, kernels, padding='same', name='block3_conv1')(x)  # Добавляем Conv2D-слой с 256-нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv1D(filters*4, kernels, padding='same', name='block3_conv2')(x)  # Добавляем Conv2D-слой с 256-нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv1D(filters*4, kernels, padding='same', name='block3_conv3')(x)  # Добавляем Conv2D-слой с 256-нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_3_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_3_out

    x = block_3_out  # Добавляем слой MaxPooling2D

    # UP 3
    x = Conv1DTranspose(filters*2, 2, strides=2, padding='same')(x)  # Добавляем слой Conv2DTranspose с 128 нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_2_out])  # Объединем текущий слой со слоем block_2_out
    x = Conv1D(filters*2, kernels, padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv1D(filters*2, kernels, padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    # UP 4
    x = Conv1DTranspose(filters, 2, strides=2, padding='same')(x)  # Добавляем слой Conv2DTranspose с 64 нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_1_out])  # Объединем текущий слой со слоем block_1_out
    x = Conv1D(filters, kernels, padding='same')(x)  # Добавляем слой Conv2D с 64 нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv1D(filters, kernels, padding='same')(x)  # Добавляем слой Conv2D с 64 нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Flatten()(x)
    x = Dense(filters*8, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(filters*2, activation='relu')(x)
    x = Dropout(0.15)(x)
    x = Dense(int(filters/2), activation='relu')(x)

    x_out = Dense(num_classes, activation='softmax')(x)

    keras_model = tf.keras.models.Model(inputs=x_in, outputs=x_out)

    return keras_model


def get_unet1d_new(
                   input_shape=(360, 15),
                   filters=64,
                   kernels=4,
                   num_classes=2,
                   ):

    x_in = Input(input_shape)  # Создаем входной слой с размерностью input_shape

    # Block 1
    x = Conv1D(filters, kernels, padding='same', name='block1_conv1')(x_in)  # Добавляем Conv2D-слой с 64-нейронами
    x = LeakyReLU(alpha=0.01)(x)  # Добавляем слой Activation

    x = Conv1D(filters, kernels, padding='same', name='block1_conv2')(x)  # Добавляем Conv2D-слой с 64-нейронами
    block_1_out = LeakyReLU(alpha=0.01)(x)  # Добавляем слой Activation и запоминаем в переменной block_1_out

    x = MaxPooling1D()(block_1_out)  # Добавляем слой MaxPooling2D

    # Block 2
    x = Conv1D(filters*2, kernels, padding='same', name='block2_conv1')(x)  # Добавляем Conv2D-слой с 128-нейронами
    x = LeakyReLU(alpha=0.01)(x)  # Добавляем слой Activation

    x = Conv1D(filters*2, kernels, padding='same', name='block2_conv2')(x)  # Добавляем Conv2D-слой с 128-нейронами
    block_2_out = LeakyReLU(alpha=0.01)(x)  # Добавляем слой Activation и запоминаем в переменной block_2_out

    x = MaxPooling1D()(block_2_out)  # Добавляем слой MaxPooling2D

    # Block 3
    x = Conv1D(filters*4, kernels, padding='same', name='block3_conv1')(x)  # Добавляем Conv2D-слой с 256-нейронами
    x = LeakyReLU(alpha=0.01)(x)  # Добавляем слой Activation

    x = Conv1D(filters*4, kernels, padding='same', name='block3_conv2')(x)  # Добавляем Conv2D-слой с 256-нейронами
    x = LeakyReLU(alpha=0.01)(x)  # Добавляем слой Activation

    x = Conv1D(filters*4, kernels, padding='same', name='block3_conv3')(x)  # Добавляем Conv2D-слой с 256-нейронами
    block_3_out = LeakyReLU(alpha=0.01)(x)  # Добавляем слой Activation и запоминаем в переменной block_3_out

    x = block_3_out  # Добавляем слой MaxPooling2D

    # UP 3
    x = Conv1DTranspose(filters*2, 2, strides=2, padding='same')(x)  # Добавляем слой Conv2DTranspose с 128 нейронами
    x = LeakyReLU(alpha=0.01)(x)  # Добавляем слой Activation

    x = concatenate([x, block_2_out])  # Объединем текущий слой со слоем block_2_out
    x = Conv1D(filters*2, kernels, padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    x = LeakyReLU(alpha=0.01)(x)  # Добавляем слой Activation

    x = Conv1D(filters*2, kernels, padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    x = LeakyReLU(alpha=0.01)(x)  # Добавляем слой Activation

    # UP 4
    x = Conv1DTranspose(filters, 2, strides=2, padding='same')(x)  # Добавляем слой Conv2DTranspose с 64 нейронами
    x = LeakyReLU(alpha=0.01)(x)  # Добавляем слой Activation

    x = concatenate([x, block_1_out])  # Объединем текущий слой со слоем block_1_out
    x = Conv1D(filters, kernels, padding='same')(x)  # Добавляем слой Conv2D с 64 нейронами
    x = LeakyReLU(alpha=0.01)(x)  # Добавляем слой Activation

    x = Conv1D(filters, kernels, padding='same')(x)  # Добавляем слой Conv2D с 64 нейронами
    x = LeakyReLU(alpha=0.01)(x)  # Добавляем слой Activation

    x = Flatten()(x)
    x = Dense(filters*8)(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(filters*2)(x)
    x = ReLU()(x)
    x = Dropout(0.15)(x)
    x = Dense(int(filters/2))(x)
    x = ReLU()(x)

    x_out = Dense(num_classes, activation='softmax')(x)

    keras_model = tf.keras.models.Model(inputs=x_in, outputs=x_out)

    return keras_model


def get_unet2d(input_shape=(88, 120, 3),
               filters=64,
               num_classes=2,
               ):

    x_in = Input(input_shape)  # Создаем входной слой с размерностью input_shape

    # Block 1
    x = Conv2D(filters, (3, 3), padding='same', name='block1_conv1')(x_in)  # Добавляем Conv2D-слой с 64-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(filters, (3, 3), padding='same', name='block1_conv2')(x)  # Добавляем Conv2D-слой с 64-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_1_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_1_out

    x = MaxPooling2D()(block_1_out)  # Добавляем слой MaxPooling2D

    # Block 2
    x = Conv2D(filters*2, (3, 3), padding='same', name='block2_conv1')(x)  # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(filters*2, (3, 3), padding='same', name='block2_conv2')(x)  # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_2_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_2_out

    x = MaxPooling2D()(block_2_out)  # Добавляем слой MaxPooling2D

    # Block 3
    x = Conv2D(filters*4, (3, 3), padding='same', name='block3_conv1')(x)  # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(filters*4, (3, 3), padding='same', name='block3_conv2')(x)  # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(filters*4, (3, 3), padding='same', name='block3_conv3')(x)  # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_3_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_3_out

    x = block_3_out  # Добавляем слой MaxPooling2D

    # UP 3
    x = Conv2DTranspose(filters*2, (2, 2), strides=(2, 2), padding='same')(
        x)  # Добавляем слой Conv2DTranspose с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_2_out])  # Объединем текущий слой со слоем block_2_out
    x = Conv2D(filters*2, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(filters*2, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    # UP 4
    x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)  # Добавляем слой Conv2DTranspose с 64 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_1_out])  # Объединем текущий слой со слоем block_1_out
    x = Conv2D(filters, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 64 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(filters, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 64 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x_out = Conv2D(num_classes, (3, 3), activation='softmax', padding='same')(x)  # Добавляем Conv2D-Слой с softmax-активацией на num_classes-нейронов

    keras_model = tf.keras.models.Model(inputs=x_in, outputs=x_out)

    return keras_model