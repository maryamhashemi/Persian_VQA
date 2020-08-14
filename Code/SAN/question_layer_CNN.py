import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, Embedding, Dropout, Concatenate, GlobalMaxPooling1D


class question_layer_CNN(Model):

    def __init__(self, num_words, embedding_dim, filter_size, num_filters, seq_length, dropout_rate, embedding_matrix, ** kwargs):

        super(question_layer_CNN, self).__init__(**kwargs)

        self.embedding = Embedding(num_words,
                                   embedding_dim,
                                   input_length=seq_length,
                                   weights=[embedding_matrix],
                                   trainable=False)
        self.conv1 = Conv1D(
            filters=num_filters[0], kernel_size=filter_size[0], activation='relu', padding='same')
        self.conv2 = Conv1D(
            filters=num_filters[1], kernel_size=filter_size[1], activation='relu', padding='same')
        self.conv3 = Conv1D(
            filters=num_filters[2], kernel_size=filter_size[2], activation='relu', padding='same')

        self.dropout = Dropout(rate=dropout_rate)
        self.maxpooling = GlobalMaxPooling1D()

    def call(self, inputs):
        # (N, SEQ_LENGTH) -> (N, SEQ_LENGTH, embedding_dim)
        x = self.embedding(inputs)

        # (N, SEQ_LENGTH, embedding_dim) -> (N, SEQ_LENGTH, NUM_FILTERS)
        h1 = self.conv1(x)
        h2 = self.conv2(x)
        h3 = self.conv3(x)

        # (N, SEQ_LENGTH, NUM_FILTERS) -> (N, NUM_FILTERS)
        h1 = self.maxpooling(h1)
        h2 = self.maxpooling(h2)
        h3 = self.maxpooling(h3)

        # (N, 1024)
        h = Concatenate(axis=1)([h1, h2, h3])

        return h
