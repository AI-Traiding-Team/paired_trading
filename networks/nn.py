import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Flatten


__version__ = 0.0002


def get_regression_model(batch_shape=(0, 299, 299, 3),
                         last_dense=512,
                         num_classes=1,
                         optimizer=tf.keras.optimizers.Adam()):

    base_model = tf.keras.applications.ResNet50V2(include_top=False,
                                                  weights=None,
                                                  input_tensor=None,
                                                  input_shape=None,
                                                  pooling=None,
                                                  # classes=classes,
                                                  # **kwargs,
                                                  )
    base_model.layers.pop(0)
    new_in = Input(batch_shape=batch_shape)
    # new_in = Flatten()(new_in)
    """  here is new 2d dimension Conv"""
    # new_in = Conv2d()(new_in)
    new_outputs = base_model(new_in)
    x = GlobalAveragePooling2D()(new_outputs)
    x = Dense(last_dense, activation='elu')(x)
    x_out = Dense(num_classes, activation='linear')(x)
    new_model = Model(inputs=new_in, outputs=x_out)

    new_model.summary()
    new_model.compile(optimizer=optimizer,
                      loss='mean_squared_error',
                      )
    return new_model


def get_classification_model(batch_shape=(0, 299, 299, 3),
                             last_dense=512,
                             num_classes=1,
                             optimizer=tf.keras.optimizers.Adam()):

    base_model = tf.keras.applications.ResNet50V2(include_top=False,
                                                  weights=None,
                                                  input_tensor=None,
                                                  input_shape=None,
                                                  pooling=None,
                                                  # classes=classes,
                                                  # **kwargs,
                                                  )
    base_model.layers.pop(0)
    new_in = Input(batch_shape=batch_shape)
    # new_in = Flatten()(new_in)
    """  here is new 2d dimension Conv"""
    # new_in = Conv2d()(new_in)
    new_outputs = base_model(new_in)
    x = GlobalAveragePooling2D()(new_outputs)
    x = Dense(last_dense, activation='elu')(x)
    x_out = Dense(num_classes, activation='linear')(x)
    new_model = Model(inputs=new_in, outputs=x_out)
    new_model.summary()
    new_model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'],
                      )
    return new_model


if __name__ == "__main__":
    # model = get_regression_model()
    model = get_classification_model(num_classes=2)

