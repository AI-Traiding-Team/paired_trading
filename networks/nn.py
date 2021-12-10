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

__version__ = 0.0014


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


@dataclass
class NNProfile:
    def __init__(self, model_type):
        self.experiment_name: str = ''
        self.experiment_directory = ''
        self.optimizer: str = "Adam"
        self.learning_rate: float = 1e-3
        self.metric: str = "mae"
        self.loss: str = 'mse'
        self.epochs: int = 50
        self.model_type: str = model_type
        self.input_shape: Tuple = None
        self.num_classes: int = 2
        self.verbose: int = 1
        self.set_type(model_type)
        pass

    def set_type(self, model_type):
        if model_type == "regression":
            self.model_type = model_type
            self.metric = 'mae'
            self.loss = "mse"
        elif model_type == "binary_crossentropy":
            self.model_type = model_type
            self.num_classes = 2
            self.metric = 'accuracy'
            self.loss = "binary_crossentropy"
        elif model_type == "categorical_crossentropy":
            self.model_type = model_type
            self.metric = 'accuracy'
            self.num_classes = 5
            self.loss = "categorical_crossentropy"
        pass


class MainNN:
    def __init__(self,
                 nn_profile: NNProfile):
        self.x_Test = None
        self.y_Test = None
        self.input_shape = None
        self.nn_profile = nn_profile
        self.history = None
        self.y_pred: np.array
        self.keras_model = tf.keras.models.Model
        self.optimizer = None
        self.dataset = DataSet

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
        elif model_type == "binary_crossentropy":
            x_out = Dense(1, activation='sigmoid')(x)
        elif model_type == "categorical_crossentropy":
            x_out = Dense(num_classes, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=x_in, outputs=x_out)
        self.nn_profile.experiment_name = f"{self.nn_profile.experiment_name}_resnet1d"
        return model

    def set_model(self):
        if self.nn_profile.optimizer == 'Adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.nn_profile.learning_rate)
        if self.nn_profile.model_type == 'regression':
            self.nn_profile.experiment_name = f"{self.nn_profile.model_type}"
            self.keras_model = self.get_resnet1d_model(input_shape=self.input_shape,
                                                       num_classes=self.nn_profile.num_classes,
                                                       model_type=self.nn_profile.model_type,
                                                       )
        elif self.nn_profile.model_type == 'binary_crossentropy':
            self.nn_profile.experiment_name = f"{self.nn_profile.model_type}"
            self.keras_model = self.get_resnet1d_model(input_shape=self.input_shape,
                                                       num_classes=self.nn_profile.num_classes,
                                                       model_type=self.nn_profile.model_type
                                                       )
        elif self.nn_profile.model_type == 'categorical_crossentropy':
            self.nn_profile.experiment_name = f"{self.nn_profile.model_type}"
            self.keras_model = self.get_resnet1d_model(input_shape=self.input_shape,
                                                       num_classes=self.nn_profile.num_classes,
                                                       model_type=self.nn_profile.model_type
                                                       )
        else:
            msg = "Error: Unknown task type for network preparation"
            sys.exit(msg)
        self.keras_model.compile(optimizer=self.optimizer,
                                 loss=self.nn_profile.loss,
                                 metrics=[self.nn_profile.metric],
                                 )
        pass

    def train_model(self, dataset: DataSet):
        self.dataset = dataset
        self.input_shape = dataset.input_shape
        self.set_model()
        path_filename = os.path.join(os.getcwd(), 'outputs', f"{self.nn_profile.experiment_name}_NN.png")
        tf.keras.utils.plot_model(self.keras_model,
                                  to_file=path_filename,
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
        path_filename = os.path.join(os.getcwd(), 'outputs', f"{self.nn_profile.experiment_name}_{self.dataset.name}.h5")
        self.keras_model.save(path_filename)
        pass

    def get_predict(self, x_Test):
        path_filename = os.path.join(os.getcwd(), 'outputs', f"{self.nn_profile.experiment_name}_{self.dataset.name}.h5")
        tf.keras.models.load_model(path_filename)
        self.y_Pred = self.keras_model.predict(x_Test)
        return self.y_Pred

    def figshow_regression(self, y_pred, y_true, delta):
        fig = plt.figure(figsize=(26, 7))
        sns.set_style("white")
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_axisbelow(True)
        ax1.minorticks_on()
        ax1.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        N = np.arange(0, len(self.history.history["loss"]))
        plt.plot(N, self.history.history["loss"], linestyle='--', color='blue', label="loss")
        if 'dice_coef' in self.history.history:
            plt.plot(N, self.history.history["dice_coef"], linestyle='--', color='green', label="dice_coef")
        if 'val_dice_coef' in self.history.history:
            plt.plot(N, self.history.history["val_dice_coef"], linestyle='-', color='red', label="val_dice_coef")
        if 'mae' in self.history.history:
            plt.plot(N, self.history.history["mae"], linestyle='--', color='red', label="mae")
        if 'accuracy' in self.history.history:
            plt.plot(N, self.history.history["accuracy"], linestyle='--', color='red', label="accuracy")
        if 'accuracy' in self.history.history:
            plt.plot(N, self.history.history["val_accuracy"], linestyle=':', color='red', label="val_accuracy")
        if 'val_loss' in self.history.history:
            plt.plot(N, self.history.history["val_loss"], linestyle=':', color='blue', label="val_accuracy")
        if 'lr' in self.history.history:
            lr_list = [x * 1000 for x in self.history.history["lr"]]
            plt.plot(N, lr_list, linestyle=':', color='green', label="lr * 1000")
        plt.title(f"Training Loss and Mean Absolute Error")
        plt.legend()
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_axisbelow(True)
        ax2.minorticks_on()
        ax2.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        plt.plot(y_pred, linestyle='--', color='red', label="Prediction")
        plt.plot(y_true, linestyle='--', color='blue', label="True")
        plt.title(f"Prediction and True ")
        plt.legend()
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_axisbelow(True)
        ax3.minorticks_on()
        ax3.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        ax3.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        # plt.plot(delta*100, linestyle='--', color='green', label="Delta percentage")
        plt.hist(delta, color='green', label="Delta percentage")
        plt.title(f"Delta percentage")
        plt.legend()
        plt.show()
        pass

    def show_regression(self):
        """
        Show evaluation for regression task

        Returns
        -------
        None
        """
        self.get_predict(self.dataset.x_Test)
        y_pred_unscaled = self.dataset.targets_scaler.inverse_transform(self.y_Pred).flatten()
        y_true_unscaled = self.dataset.targets_scaler.inverse_transform(self.dataset.y_Test).flatten()

        # вычисление среднего значения, средней ошибки и процента ошибки
        mean_value = sum(y_pred_unscaled) / len(y_pred_unscaled)
        delta = abs(y_pred_unscaled - y_true_unscaled)
        delta_percentage = delta / y_true_unscaled
        self.figshow_regression(y_pred_unscaled, y_true_unscaled, delta_percentage)
        mean_delta = sum(delta) / len(delta)
        mean_value_info = f"Среднее значение: {round(mean_value, 2)} \n"
        mean_delta_info = f"Средняя ошибка: {round(mean_delta, 2)} \n"
        mean_percent_info = f"Средний процент ошибки: {round(100 * mean_delta / mean_value, 2)}%"
        text_data = mean_value_info + mean_delta_info + mean_percent_info
        print(text_data)

    def figshow_base(self):
        fig = plt.figure(figsize=(26, 7))
        sns.set_style("white")
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_axisbelow(True)
        ax1.minorticks_on()
        ax1.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        N = np.arange(0, len(self.history.history["loss"]))
        plt.plot(N, self.history.history["loss"], linestyle='--', color='blue', label="loss")
        if 'dice_coef' in self.history.history:
            plt.plot(N, self.history.history["dice_coef"], linestyle='--', color='green', label="dice_coef")
        if 'val_dice_coef' in self.history.history:
            plt.plot(N, self.history.history["val_dice_coef"], linestyle='-', color='red', label="val_dice_coef")
        if 'mae' in self.history.history:
            plt.plot(N, self.history.history["mae"], linestyle='--', color='red', label="mae")
        if 'accuracy' in self.history.history:
            plt.plot(N, self.history.history["accuracy"], linestyle='--', color='red', label="accuracy")
        if 'accuracy' in self.history.history:
            plt.plot(N, self.history.history["val_accuracy"], linestyle=':', color='red', label="val_accuracy")
        if 'val_loss' in self.history.history:
            plt.plot(N, self.history.history["val_loss"], linestyle=':', color='blue', label="val_accuracy")
        if 'lr' in self.history.history:
            lr_list = [x * 1000 for x in self.history.history["lr"]]
            plt.plot(N, lr_list, linestyle=':', color='green', label="lr * 1000")
        plt.title(f"Training Loss and Mean Absolute Error")
        plt.legend()
        # ax2 = fig.add_subplot(1, 3, 2)
        # ax2.set_axisbelow(True)
        # ax2.minorticks_on()
        # ax2.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        # ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        # plt.plot(y_pred, linestyle='--', color='red', label="Prediction")
        # plt.plot(y_true, linestyle='--', color='blue', label="True")
        # plt.title(f"Prediction and True ")
        # plt.legend()
        # ax3 = fig.add_subplot(1, 3, 3)
        # ax3.set_axisbelow(True)
        # ax3.minorticks_on()
        # ax3.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        # ax3.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        # # plt.plot(delta*100, linestyle='--', color='green', label="Delta percentage")
        # plt.hist(delta, color='green', label="Delta percentage")
        # plt.title(f"Delta percentage")
        # plt.legend()
        plt.show()
        pass

    def check_categorical(self):
        x_test = self.dataset.x_Test[-600:-500]
        y_test_org = self.dataset.y_Test[-600:-500]
        conv_test = []
        for i in range(len(x_test)):
            x = x_test[i]
            x = np.expand_dims(x, axis=0)
            prediction = self.keras_model.predict(x)  # Распознаём наш пример
            # print('\n',prediction)
            prediction = np.argmax(prediction)  # Получаем индекс самого большого элемента (это итоговая цифра)
            if prediction == np.argmax(y_test_org[i]):
                conv_test.append('True')
            else:
                conv_test.append('False')

            print('Index:', i, '\tPrediction:', prediction, 'Real:', np.argmax(y_test_org[i]),
                  '\t====>', y_test_org[i])
            pass

    def show_categorical(self):
        """
        Show evaluation for categorical classification task

        Returns
        -------
        None
        """
        self.get_predict(self.dataset.x_Test)
        y_pred_unscaled = self.y_Pred.flatten()
        y_true_unscaled = self.dataset.y_Test.flatten()

        # вычисление среднего значения, средней ошибки и процента ошибки
        # mean_value = sum(y_pred_unscaled) / len(y_pred_unscaled)
        # delta = abs(y_pred_unscaled - y_true_unscaled)
        # delta_percentage = delta / y_true_unscaled
        self.figshow_base()
        self.check_categorical()

        pass

if __name__ == "__main__":
    test_nn_profile = NNProfile()
    test_nn_profile.experiment_name = "regression_resnet1d_close1_close2"
    test_nn = MainNN(test_nn_profile)
