import keras
import tensorflow as tf
from keras import backend as K
from keras import layers


class GeM(layers.Layer):
    def __init__(self, pool_size, init_norm=3.0, normalize=False, **kwargs):
        self.pool_size = pool_size
        self.init_norm = init_norm
        self.normalize = normalize

        super(GeM, self).__init__(**kwargs)

    def build(self, input_shape):
        feature_size = input_shape[-1]
        self.p = self.add_weight(name="norms", shape=(feature_size,),
                                 initializer=keras.initializers.constant(self.init_norm),
                                 trainable=True)
        super(GeM, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        x = K.maximum(x, 1e-6)
        x = K.pow(x, self.p)

        x = K.pool2d(x, pool_size=(self.pool_size, self.pool_size), strides=(self.pool_size, self.pool_size),
                     pool_mode='avg', padding='valid')
        x = K.pow(x, 1.0 / self.p)

        if self.normalize:
            x = K.l2_normalize(x, 1)
        return x

    def compute_output_shape(self, input_shape):
        return tuple([None, input_shape[-1]])
