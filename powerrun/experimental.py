from tensorflow.keras.layers import Dense, Input, Flatten, Conv1D, ReLU, ELU, LeakyReLU, MaxPool1D, AveragePooling1D, Dropout, \
    Conv2D, LeakyReLU, Reshape, BatchNormalization, Activation, Conv2DTranspose, Conv1DTranspose, Conv1D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, MaxPooling1D, concatenate, GlobalAveragePooling2D, \
    GlobalAveragePooling1D, GlobalMaxPooling1D, LSTM, GRU, MultiHeadAttention, LayerNormalization
from tensorflow.keras.activations import tanh
import tensorflow as tf


__version__ = 0.0004


def get_unet1d_moded(input_shape=(88, 120, 1),
                     filters=64,
                     kernels=5,
                     num_classes=2,
                     ):

    x_in = Input(input_shape)  # Создаем входной слой с размерностью input_shape

    # Block 1
    x = Conv1D(filters, kernels, padding='same', name='block1_conv1')(x_in)  # Добавляем Conv2D-слой с 64-нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('elu')(x)  # Добавляем слой Activation

    x = Conv1D(filters, kernels, padding='same', name='block1_conv2')(x)  # Добавляем Conv2D-слой с 64-нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_1_out = Activation('elu')(x)  # Добавляем слой Activation и запоминаем в переменной block_1_out

    x = MaxPooling1D()(block_1_out)  # Добавляем слой MaxPooling2D

    # Block 2
    x = Conv1D(filters*2, kernels, padding='same', name='block2_conv1')(x)  # Добавляем Conv2D-слой с 128-нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('elu')(x)  # Добавляем слой Activation

    x = Conv1D(filters*2, kernels, padding='same', name='block2_conv2')(x)  # Добавляем Conv2D-слой с 128-нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_2_out = Activation('elu')(x)  # Добавляем слой Activation и запоминаем в переменной block_2_out

    x = MaxPooling1D()(block_2_out)  # Добавляем слой MaxPooling2D

    # Block 3
    x = Conv1D(filters*4, kernels, padding='same', name='block3_conv1')(x)  # Добавляем Conv2D-слой с 256-нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('elu')(x)  # Добавляем слой Activation

    x = Conv1D(filters*4, kernels, padding='same', name='block3_conv2')(x)  # Добавляем Conv2D-слой с 256-нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('elu')(x)  # Добавляем слой Activation

    x = Conv1D(filters*4, kernels, padding='same', name='block3_conv3')(x)  # Добавляем Conv2D-слой с 256-нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_3_out = Activation('elu')(x)  # Добавляем слой Activation и запоминаем в переменной block_3_out

    x = MaxPooling1D()(block_3_out)  # Добавляем слой MaxPooling2D

    # Block 4
    x = Conv1D(filters*8, kernels, padding='same', name='block4_conv1')(x)
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('elu')(x)  # Добавляем слой Activation

    x = Conv1D(filters*8, kernels, padding='same', name='block4_conv2')(x)
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('elu')(x)  # Добавляем слой Activation

    x = Conv1D(filters*8, kernels, padding='same', name='block4_conv3')(x)
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_4_out = Activation('elu')(x)

    x = block_4_out  # Добавляем слой MaxPooling2D

    # UP 4
    x = Conv1DTranspose(filters*4, 2, strides=2, padding='same')(x)
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('elu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_3_out])  # Объединем текущий слой со слоем block_2_out
    x = Conv1D(filters*4, kernels, padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('elu')(x)  # Добавляем слой Activation

    x = Conv1D(filters*4, kernels, padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('elu')(x)  # Добавляем слой Activation

    # UP 3
    x = Conv1DTranspose(filters*2, 2, strides=2, padding='same')(
        x)  # Добавляем слой Conv2DTranspose с 128 нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('elu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_2_out])  # Объединем текущий слой со слоем block_2_out
    x = Conv1D(filters*2, kernels, padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('elu')(x)  # Добавляем слой Activation

    x = Conv1D(filters*2, kernels, padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('elu')(x)  # Добавляем слой Activation

    # UP 4
    x = Conv1DTranspose(filters, kernels, strides=2, padding='same')(x)  # Добавляем слой Conv2DTranspose с 64 нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('elu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_1_out])  # Объединем текущий слой со слоем block_1_out
    x = Conv1D(filters, kernels, padding='same')(x)  # Добавляем слой Conv2D с 64 нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('elu')(x)  # Добавляем слой Activation

    x = Conv1D(filters, kernels, padding='same')(x)  # Добавляем слой Conv2D с 64 нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('elu')(x)  # Добавляем слой Activation

    x = Flatten()(x)
    x = Dense(filters*8, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(filters*2, activation='relu')(x)
    x = Dropout(0.15)(x)
    x = Dense(int(filters/2), activation='relu')(x)

    x_out = Dense(num_classes, activation='softmax')(x)

    keras_model = tf.keras.models.Model(inputs=x_in, outputs=x_out)

    return keras_model


def get_unet1d_tanh(
                    input_shape=(88, 120, 1),
                    filters=64,
                    kernels=8,
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


def get_3way_lstm_moded(input_shape=(360, 15),
                        filters=15,
                        kernels_len=15,
                        num_classes=2,
                        ):

    twin_filters = input_shape[1]*4
    conv_kernel_size = int(kernels_len/3)

    x_in = Input(shape=input_shape)
    lstmWay = LSTM(twin_filters, return_sequences="True")(x_in)
    convWay = Conv1D(filters, kernel_size=conv_kernel_size, activation='elu', padding='same')(x_in)

    lstmConvWay = Conv1D(filters,  kernel_size=conv_kernel_size, activation='elu', padding='same')(lstmWay)
    convLstmWay = LSTM(twin_filters, return_sequences="True")(convWay)

    lstmWay = LSTM(twin_filters, return_sequences="True")(lstmWay)
    convWay = Conv1D(filters, kernel_size=conv_kernel_size, activation='elu', padding='same')(convWay)

    lstmWay = Flatten()(lstmWay)
    convWay = Flatten()(convWay)
    lstmConvWay = Flatten()(lstmConvWay)
    convLstmWay = Flatten()(convLstmWay)

    finWay = concatenate([lstmWay, convWay, lstmConvWay, convLstmWay])
    x = Dropout(0.5)(finWay)
    x = Dense(twin_filters*4)(x)
    x = ReLU()(x)
    x = Dropout(0.15)(x)
    x = Dense(twin_filters)(x)
    x = ReLU()(x)
    x_out = Dense(num_classes, activation="softmax")(x)

    keras_model = tf.keras.models.Model(inputs=x_in, outputs=x_out)

    return keras_model


""" Transformer """


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_model(
                input_shape,
                num_classes,
                head_size,
                num_heads,
                ff_dim,
                num_transformer_blocks,
                mlp_units,
                dropout=0,
                mlp_dropout=0,
                ):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)

