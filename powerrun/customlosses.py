import tensorflow as tf
from tensorflow.keras import backend
import tensorflow.keras.metrics

__version__ = 0.10


class DiceCoefficient(tensorflow.keras.metrics.Metric):

    def __init__(self, name='dice_coef', **kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.dice: float = 0
        self.short_name = name
        pass

    def update_state(self, y_true, y_pred, smooth=1, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # по размеру тензора определяем пришли маски по изображениям или по тексту
        shape = y_true.shape
        shape_len = len(shape)
        if shape_len == 2:
            axis = [1]
        elif shape_len == 3:
            axis = [1, 2]
        elif shape_len == 4:
            axis = [1, 2, 3]
        intersection = backend.sum(y_true * y_pred, axis=axis)
        union = backend.sum(y_true, axis=axis) + backend.sum(y_pred, axis=axis)
        self.dice = backend.mean((2. * intersection + smooth) / (union + smooth), axis=0)

    def result(self):
        return self.dice

    def __str__(self):
        return self.short_name

    def reset_states(self):
        self.dice: float = 0
        pass


class DiceCCELoss(tensorflow.keras.metrics.Metric):
    def __init__(self, name='dice_cce_loss', **kwargs):
        super(DiceCCELoss, self).__init__(name=name, **kwargs)
        self.dice: float = 0
        self.cce_loss: float = 0
        self.short_name = name
        self.cce = tf.keras.losses.CategoricalCrossentropy()
        pass

    def update_state(self, y_true, y_pred, smooth=1, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # по размеру тензора определяем пришли маски по изображениям или по тексту
        shape = y_true.shape
        shape_len = len(shape)
        if shape_len == 2:
            axis = [1]
        elif shape_len == 3:
            axis = [1, 2]
        elif shape_len == 4:
            axis = [1, 2, 3]
        self.cce_loss = self.cce(y_true, y_pred)
        intersection = backend.sum(y_true * y_pred, axis=axis)
        union = backend.sum(y_true, axis=axis) + backend.sum(y_pred, axis=axis)
        self.dice = 1 - backend.mean((2. * intersection + smooth) / (union + smooth), axis=0)
        self.dice = self.cce_loss + self.dice

    def result(self):
        return self.dice

    def __str__(self):
        return self.short_name

    def reset_states(self):
        self.dice: float = 0
        pass