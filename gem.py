import tensorflow as tf
from keras import layers
import keras

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
        x = tf.math.max(x, 1e-6)
        x = tf.pow(x, self.p)

        x = tf.nn.avg_pool(
            x,
            self.pool_size,
            self.pool_size,
            'VALID',
        )
        x = tf.pow(x, 1.0 / self.p)

        if normalize:
            x = tf.nn.l2_normalize(x, 1)
        return x

    def compute_output_shape(self, input_shape):
        return tuple([None, input_shape[-1]])
