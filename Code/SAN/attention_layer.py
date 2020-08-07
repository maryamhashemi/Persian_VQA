import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout


class attention_layer(Model):

    def __init__(self, attention_dim, **kwargs):
        super(attention_layer, self).__init__(**kwargs)
        self.wi = Dense(units=attention_dim, use_bias=False, activation=None)
        self.wq = Dense(units=attention_dim, activation=None)
        self.wpi = Dense(1, activation='softmax')
        self.dropout = Dropout(0.5)

    def call(self, inputs):
        vi, vq = inputs

        # (N, 196, 1024) -> (N, 196, attention_dim)
        hi = self.wi(vi)

        # (N, 1024) -> (N, attention_dim)
        hq = self.wq(vq)

        # (N, attention_dim) -> (N, 1, attention_dim)
        hq = tf.expand_dims(hq, axis=1)

        # (N, 196, attention_dim)
        ha = tf.tanh(hi + hq)

        ha = self.dropout(ha)

        # (N, 196, attention_dim) -> (N, 196,  1)
        pi = self.wpi(ha)

        # (N, 196, 1), (N, 196, 1024) -> (N, 1024)
        vi_att = tf.reduce_sum(pi*vi, axis=1)

        # (N, 1024)
        u = vi_att + vq

        return u
