from datamodeling import *
from analyze import DataLoad
from networks import *


def get_resnet1d_model(
                       input_shape=(40, 15,),
                       num_classes=2,
                       kernels=32,
                       stride=3,
                       ):
    def residual_block(x, kernels, stride):
        tf.keras.backend.set_floatx('float64')
        out = Conv1D(kernels, stride, padding='same')(x)
        out = ReLU()(out)
        out = Conv1D(kernels, stride, padding='same')(out)
        out = tf.keras.layers.add([x, out])
        out = ReLU()(out)
        out = MaxPool1D(3, 2)(out)
        return out

    activation_1 = 'relu'

    x_in = Input(shape=input_shape)
    x = Conv1D(kernels, stride)(x_in)
    x = residual_block(x, kernels, stride)
    x = residual_block(x, kernels, stride)
    x = residual_block(x, kernels, stride)
    x = residual_block(x, kernels, stride)
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x_out = Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=x_in, outputs=x_out)
    return model

loaded_crypto_data = DataLoad(pairs_symbols=None,
                              time_intervals=['1m'],
                              source_directory="../source_root",
                              start_period='2021-09-01 00:00:00',
                              end_period='2021-12-05 23:59:59',
                              )

dataset_profile = DSProfile()
# self.dataset_profile.Y_data = "power_trend_binary"
dataset_profile.Y_data = "power_trend_binary"
dataset_profile.timeframe = "1m"
dataset_profile.use_symbols_pairs = ("ETHUSDT", "BTCUSDT", "ETHBTC")
dataset_profile.power_trend = 0.075

""" Default options for dataset window"""
dataset_profile.tsg_window_length = 40
dataset_profile.tsg_sampling_rate = 1
dataset_profile.tsg_stride = 1
dataset_profile.tsg_start_index = 0

""" Warning! Change this qty if using .shift() more then 2 """
dataset_profile.tsg_overlap = 0
dsc = DSCreator(loaded_crypto_data, dataset_profile)

nn_profile = NNProfile("categorical_crossentropy")
nn_profile.learning_rate = 1e-4
nn_profile.experiment_name = f"{nn_profile.experiment_name}_categorical_trend"
nn_profile.epochs = 10
nn_network = MainNN(nn_profile)
nn_network.keras_model = get_resnet1d_model()
nn_network.keras_model.compile(optimizer=nn_network.optimizer,
                               loss=nn_profile.loss,
                               metrics=[nn_profile.metric],
                              )

dts_power_trend = dsc.create_dataset()
nn_network.train_model(dts_power_trend)
# test_nn.get_predict()
# test_nn.show_regression()
