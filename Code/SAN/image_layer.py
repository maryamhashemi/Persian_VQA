import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense


class image_layer(Model):

    def __init__(self, **kwargs):
        super(image_layer, self).__init__(**kwargs)
        self.dense = Dense(1024, activation='tanh')

    def call(self, inputs):
        # N * 512 * 14 * 14 -> N * 512 * 196
        x = tf.reshape(inputs, [-1, inputs.shape[1],
                                inputs.shape[2]*inputs.shape[3]])

        # N * 512 * 196 -> N * 196 * 512
        x = tf.transpose(x, perm=[0, 2, 1])

        # N * 196 * 512 -> N * 196 * 1024
        x = self.dense(x)

        return x
