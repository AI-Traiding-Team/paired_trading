import os
import sys
from typing import Tuple

import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Flatten, Conv1D, ReLU, ELU, MaxPool1D
from datamodeling.dscreator import DataSet
import matplotlib.pyplot as plt
import seaborn as sns

__version__ = 0.0008


# def get_regression_model(batch_shape=(0, 299, 299, 3),
#                          last_dense=512,
#                          num_classes=1,
#                          optimizer=tf.keras.optimizers.Adam()):
#     base_model = tf.keras.applications.ResNet50V2(include_top=False,
#                                                   weights=None,
#                                                   input_tensor=None,
#                                                   input_shape=None,
#                                                   pooling=None,
#                                                   # classes=classes,
#                                                   # **kwargs,
#                                                   )
#     base_model.layers.pop(0)
#     new_in = Input(batch_shape=batch_shape)
#     # new_in = Flatten()(new_in)
#     """  here is new 2d dimension Conv"""
#     # new_in = Conv2d()(new_in)
#     new_outputs = base_model(new_in)
#     x = GlobalAveragePooling2D()(new_outputs)
#     x = Dense(last_dense, activation='elu')(x)
#     x_out = Dense(num_classes, activation='linear')(x)
#     new_model = Model(inputs=new_in, outputs=x_out)
#
#     new_model.summary()
#     new_model.compile(optimizer=optimizer,
#                       loss='mean_squared_error',
#                       )
#     return new_model


# def get_classification_model(batch_shape=(0, 299, 299, 3),
#                              last_dense=512,
#                              num_classes=1,
#                              optimizer=tf.keras.optimizers.Adam()):
#     base_model = tf.keras.applications.ResNet50V2(include_top=False,
#                                                   weights=None,
#                                                   input_tensor=None,
#                                                   input_shape=None,
#                                                   pooling=None,
#                                                   # classes=classes,
#                                                   # **kwargs,
#                                                   )
#     base_model.layers.pop(0)
#     new_in = Input(batch_shape=batch_shape)
#     new_in = Flatten()(new_in)
#     new_in = Dense(batch_shape[1*4], activation='elu')(new_in)
#     new_in = Dense(batch_shape[1*4], activation='elu')(new_in)
#     """  here is new 2d dimension Conv"""
#     # new_in = Conv2d()(new_in)
#     new_outputs = base_model(new_in)
#     x = GlobalAveragePooling2D()(new_outputs)
#     x = Dense(last_dense, activation='elu')(x)
#     x_out = Dense(num_classes, activation='linear')(x)
#     new_model = Model(inputs=new_in, outputs=x_out)
#     new_model.summary()
#     new_model.compile(optimizer=optimizer,
#                       loss='binary_crossentropy',
#                       metrics=['accuracy'],
#                       )
#     return new_model


def get_resnet_model(input_shape=(40, 15, ),
                     num_classes=2,
                     kernels=32,
                     # stride=5,
                     stride=3,
                     model_type="regression",
                     ):

  def residual_block(x, kernels, stride):
    out = Conv1D(kernels, stride, padding='same')(x)
    out = ReLU()(out)
    out = Conv1D(kernels, stride, padding='same')(out)
    out = tf.keras.layers.add([x, out])
    out = ReLU()(out)
    out = MaxPool1D(3, 2)(out)
    return out

  activation_1 = 'relu'
  if model_type == 'regression':
      activation_1 = 'elu'

  x_in = Input(shape=input_shape)
  x = Conv1D(kernels, stride)(x_in)
  x = residual_block(x, kernels, stride)
  x = residual_block(x, kernels, stride)
  x = residual_block(x, kernels, stride)
  x = residual_block(x, kernels, stride)
  # x = residual_block(x, kernels, stride)
  x = Flatten()(x)
  x = Dense(32, activation=activation_1)(x)
  x = Dense(32, activation=activation_1)(x)
  # x_out = (Dense(1, activation='sigmoid')(x) if num_classes == 2 else

  if model_type == 'regression':
      x_out = Dense(1, activation='linear')(x)
  elif model_type == "binary_classification":
      x_out = Dense(1, activation='sigmoid')(x)
  elif model_type == "classification":
      x_out = Dense(num_classes, activation='softmax')(x)
  model = tf.keras.models.Model(inputs=x_in, outputs=x_out)
  return model


@dataclass(init=True)
class NNProfile:
    experiment_name: str = ''
    experiment_directory = ''
    optimizer: str = "Adam"
    learning_rate: float = 1e-3
    metric: str  = "mae"
    loss: str = 'binary_crossentropy'
    epochs: int = 50
    model_type: str = 'regression'
    input_shape: Tuple = None
    num_classes: int = 2
    verbose: int = 1
    pass


class MainNN:
    def __init__(self,
                 nn_profile: NNProfile):
        self.input_shape = None
        self.nn_profile = nn_profile
        self.history = None
        self.keras_model = tf.keras.models.Model
        self.optimizer = None

    def get_resnet1d_model(self,
                           input_shape=(40, 15,),
                           num_classes=2,
                           kernels=32,
                           stride=3,
                           model_type="regression",
                           ):

        def residual_block(x, kernels, stride):
            out = Conv1D(kernels, stride, padding='same')(x)
            out = ReLU()(out)
            out = Conv1D(kernels, stride, padding='same')(out)
            out = tf.keras.layers.add([x, out])
            out = ReLU()(out)
            out = MaxPool1D(3, 2)(out)
            return out

        activation_1 = 'relu'
        if model_type == 'regression':
            activation_1 = 'elu'

        x_in = Input(shape=input_shape)
        x = Conv1D(kernels, stride)(x_in)
        x = residual_block(x, kernels, stride)
        x = residual_block(x, kernels, stride)
        x = residual_block(x, kernels, stride)
        x = residual_block(x, kernels, stride)
        x = Flatten()(x)
        x = Dense(32, activation=activation_1)(x)
        x = Dense(32, activation=activation_1)(x)

        if model_type == 'regression':
            x_out = Dense(1, activation='linear')(x)
        elif model_type == "binary_classification":
            x_out = Dense(1, activation='sigmoid')(x)
        elif model_type == "classification":
            x_out = Dense(num_classes, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=x_in, outputs=x_out)
        return model

    def set_model(self):
        if self.nn_profile.optimizer =='Adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.nn_profile.learning_rate)
        if self.nn_profile.model_type == 'regression':
            self.keras_model = self.get_resnet1d_model(input_shape=self.input_shape,
                                                      num_classes=self.nn_profile.num_classes
                                                      )
            self.keras_model.compile(optimizer=self.optimizer,
                                     loss='mse',
                                     metrics=['mae'],
                                     )
        pass

    def train_model(self, dataset: DataSet):
        self.input_shape = dataset.input_shape
        self.set_model()
        tf.keras.utils.plot_model(self.keras_model,
                                  to_file=f'outputs/{self.nn_profile.experiment_name}_NN.png',
                                  show_shapes=True,
                                  show_layer_names=True,
                                  expand_nested=True,
                                  dpi=96,
                                  )

        self.history = self.keras_model.fit(dataset.train_gen,
                                            epochs=self.nn_profile.epochs,
                                            validation_data=dataset.val_gen,
                                            verbose=self.nn_profile.verbose,
                                            )
        self.keras_model.save(os.path.join('outputs/', f"{self.nn_profile.experiment_name}.h5"))
        pass



if __name__ == "__main__":
    # model = get_regression_model()
    test_nn_profile = NNProfile()
    test_nn_profile.experiment_name = "test_NN_regression_ResNetV2"
    test_nn = MainNN(test_nn_profile)
