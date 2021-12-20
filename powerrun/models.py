from tensorflow.keras.layers import Dense, Input, Flatten, Conv1D, ReLU, ELU, MaxPool1D, AveragePooling1D, Dropout, \
    Conv2D, LeakyReLU, Reshape, BatchNormalization, Activation, Conv2DTranspose, Conv1DTranspose, Conv1D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, MaxPooling1D, concatenate, GlobalAveragePooling2D, \
    GlobalAveragePooling1D, GlobalMaxPooling1D
import tensorflow as tf

__version__ = 0.00012


def get_resnet50v2_classification_model(input_shape=(40, 16, 1),
                                        kernels=64,
                                        num_classes=2,
                                        ):

    input_shape = (*input_shape, 1)
    new_in = Input(shape=input_shape)
    # heading = Reshape((int(input_shape[0]/2), input_shape[1]*2, 1,))(new_in)

    base_model = tf.keras.applications.ResNet50V2(include_top=False,
                                                  weights=None,
                                                  input_tensor=None,
                                                  # input_shape=(int(input_shape[0]/2), input_shape[1]*2, 1,),
                                                  input_shape=input_shape,
                                                  pooling=None,
                                                  classes=num_classes,
                                                  # **kwargs,
                                                  )
    base_model.layers.pop(0)
    base_model.trainable = True
    # base_output = base_model(heading)
    base_output = base_model(new_in)
    x = GlobalAveragePooling2D()(base_output)
    x = Dense(kernels*4, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(kernels*2, activation='relu')(x)
    x = Dropout(0.35)(x)
    x = Dense(int(kernels/2), activation='relu')(x)
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
                            input_shape=(40, 16,),
                            kernels=64,
                            stride=4,
                            ):
    # tf.keras.backend.set_floatx('float64')

    def residual_block_max(x, kernels, stride, pool=2):
        out = Conv1D(kernels, stride, padding='same')(x)
        out = ReLU()(out)
        out = Conv1D(kernels, stride, padding='same')(out)
        out = tf.keras.layers.add([x, out])
        out = ReLU()(out)
        out = MaxPool1D(pool, 2)(out)
        return out

    x_in = Input(shape=input_shape)
    x = Dense(kernels, activation="relu")(x_in)
    x = Conv1D(kernels, stride)(x)
    x1_1 = residual_block_max(x, kernels, stride, pool=10)
    x1_2 = residual_block_max(x1_1, kernels, stride, pool=9)
    x1_3 = residual_block_max(x1_2, kernels, stride, pool=5)
    x1_4 = residual_block_max(x1_3, kernels, stride, pool=4)
    x1_5 = residual_block_max(x1_4, kernels, stride, pool=3)
    x1_6 = residual_block_max(x1_5, kernels, stride, pool=2)

    x = concatenate([x, x1_1, x1_2, x1_3, x1_4, x1_5, x1_6], axis=-2)
    x = Flatten()(x)
    x = Dense(kernels*4, activation="relu")(x)
    x = Dropout(0.35)(x)
    x = Dense(kernels*2, activation="relu")(x)
    x = Dropout(0.35)(x)
    x = Dense(32, activation="relu")(x)
    x_out = Dense(2, activation='softmax')(x)
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


def get_unet1d(input_shape=(88, 120, 1),
               kernels=64,
               num_classes=2,
               ):

    x_in = Input(input_shape)  # Создаем входной слой с размерностью input_shape

    # Block 1
    x = Conv1D(kernels, 4, padding='same', name='block1_conv1')(x_in)  # Добавляем Conv2D-слой с 64-нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('elu')(x)  # Добавляем слой Activation

    x = Conv1D(kernels, 4, padding='same', name='block1_conv2')(x)  # Добавляем Conv2D-слой с 64-нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_1_out = Activation('elu')(x)  # Добавляем слой Activation и запоминаем в переменной block_1_out

    x = MaxPooling1D()(block_1_out)  # Добавляем слой MaxPooling2D

    # Block 2
    x = Conv1D(kernels*2, 4, padding='same', name='block2_conv1')(x)  # Добавляем Conv2D-слой с 128-нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('elu')(x)  # Добавляем слой Activation

    x = Conv1D(kernels*2, 4, padding='same', name='block2_conv2')(x)  # Добавляем Conv2D-слой с 128-нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_2_out = Activation('elu')(x)  # Добавляем слой Activation и запоминаем в переменной block_2_out

    x = MaxPooling1D()(block_2_out)  # Добавляем слой MaxPooling2D

    # Block 3
    x = Conv1D(kernels*4, 4, padding='same', name='block3_conv1')(x)  # Добавляем Conv2D-слой с 256-нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('elu')(x)  # Добавляем слой Activation

    x = Conv1D(kernels*4, 4, padding='same', name='block3_conv2')(x)  # Добавляем Conv2D-слой с 256-нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('elu')(x)  # Добавляем слой Activation

    x = Conv1D(kernels*4, 4, padding='same', name='block3_conv3')(x)  # Добавляем Conv2D-слой с 256-нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_3_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_3_out

    x = block_3_out  # Добавляем слой MaxPooling2D

    # UP 3
    x = Conv1DTranspose(kernels*2, 2, strides=2, padding='same')(
        x)  # Добавляем слой Conv2DTranspose с 128 нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('elu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_2_out])  # Объединем текущий слой со слоем block_2_out
    x = Conv1D(kernels*2, 4, padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('elu')(x)  # Добавляем слой Activation

    x = Conv1D(kernels*2, 4, padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('elu')(x)  # Добавляем слой Activation

    # UP 4
    x = Conv1DTranspose(kernels, 2, strides=2, padding='same')(x)  # Добавляем слой Conv2DTranspose с 64 нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('elu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_1_out])  # Объединем текущий слой со слоем block_1_out
    x = Conv1D(kernels, 4, padding='same')(x)  # Добавляем слой Conv2D с 64 нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('elu')(x)  # Добавляем слой Activation

    x = Conv1D(kernels, 4, padding='same')(x)  # Добавляем слой Conv2D с 64 нейронами
    # x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('elu')(x)  # Добавляем слой Activation

    x = Flatten()(x)
    x = Dense(kernels*8, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(kernels*2, activation='relu')(x)
    x = Dropout(0.15)(x)
    x = Dense(int(kernels/2), activation='relu')(x)

    x_out = Dense(2, activation='softmax')(x)

    keras_model = tf.keras.models.Model(inputs=x_in, outputs=x_out)

    return keras_model


def get_unet2d (input_shape=(88, 120, 3),
                kernels=64,
                num_classes=2,
                ):

    x_in = Input(input_shape)  # Создаем входной слой с размерностью input_shape

    # Block 1
    x = Conv2D(kernels, (3, 3), padding='same', name='block1_conv1')(x_in)  # Добавляем Conv2D-слой с 64-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(kernels, (3, 3), padding='same', name='block1_conv2')(x)  # Добавляем Conv2D-слой с 64-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_1_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_1_out

    x = MaxPooling2D()(block_1_out)  # Добавляем слой MaxPooling2D

    # Block 2
    x = Conv2D(kernels*2, (3, 3), padding='same', name='block2_conv1')(x)  # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(kernels*2, (3, 3), padding='same', name='block2_conv2')(x)  # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_2_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_2_out

    x = MaxPooling2D()(block_2_out)  # Добавляем слой MaxPooling2D

    # Block 3
    x = Conv2D(kernels*4, (3, 3), padding='same', name='block3_conv1')(x)  # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(kernels*4, (3, 3), padding='same', name='block3_conv2')(x)  # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(kernels*4, (3, 3), padding='same', name='block3_conv3')(x)  # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_3_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_3_out

    x = block_3_out  # Добавляем слой MaxPooling2D

    # UP 3
    x = Conv2DTranspose(kernels*2, (2, 2), strides=(2, 2), padding='same')(
        x)  # Добавляем слой Conv2DTranspose с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_2_out])  # Объединем текущий слой со слоем block_2_out
    x = Conv2D(kernels*2, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(kernels*2, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    # UP 4
    x = Conv2DTranspose(kernels, (2, 2), strides=(2, 2), padding='same')(x)  # Добавляем слой Conv2DTranspose с 64 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_1_out])  # Объединем текущий слой со слоем block_1_out
    x = Conv2D(kernels, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 64 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(kernels, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 64 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x_out = Conv2D(num_classes, (3, 3), activation='softmax', padding='same')(x)  # Добавляем Conv2D-Слой с softmax-активацией на num_classes-нейронов

    keras_model = tf.keras.models.Model(inputs=x_in, outputs=x_out)

    return keras_model