import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPool2D, LSTM, Activation, Embedding
from tensorflow.keras.initializers import glorot_normal, RandomNormal, he_normal, glorot_uniform, orthogonal
from constants import *


class CoattentionModel(Layer):

    def __init__(self):
        super().__init__()
        self.num_classes = NUM_CLASSES
        self.hidden_size = 512
        self.dropout = 0.5
        self.num_embeddings = EMBEDDING_DIM

        self.image_dense = Dense(self.hidden_size,
                                 kernel_initializer=glorot_normal(seed=15))
        self.image_corr = Dense(self.hidden_size,
                                kernel_initializer=glorot_normal(seed=29))

        self.image_atten_dense = Dense(self.hidden_size,
                                       kernel_initializer=glorot_uniform(seed=17))
        self.question_atten_dens = Dense(self.hidden_size,
                                         kernel_initializer=glorot_uniform(seed=28))

        self.question_atten_dropout = Dropout(self.dropout)
        self.image_atten_dropout = Dropout(self.dropout)

        self.ques_atten = Dense(1, kernel_initializer=glorot_uniform(seed=21))
        self.img_atten = Dense(1, kernel_initializer=glorot_uniform(seed=33))

        self.embed = Embedding(self.num_embeddings,
                               self.hidden_size,
                               embeddings_initializer=RandomNormal(mean=0, stddev=1, seed=23))

        self.unigram_conv = Conv1D(filters=self.hidden_size,
                                   kernel_size=1,
                                   strides=1,
                                   padding='same',
                                   kernel_initializer=glorot_normal(seed=41))
        self.bigram_conv = Conv1D(filters=self.hidden_size,
                                  kernel_size=2,
                                  strides=1,
                                  padding='same',
                                  kernel_initializer=glorot_normal(seed=58), dilation_rate=2)
        self.trigram_conv = Conv1D(filters=self.hidden_size,
                                   kernel_size=3,
                                   strides=1,
                                   padding='same',
                                   kernel_initializer=glorot_normal(seed=89), dilation_rate=2)

        self.max_pool = MaxPool2D((3, 1))
        self.phrase_dropout = Dropout(self.dropout)

        self.lstm = LSTM(units=512,
                         return_sequences=True,
                         dropout=self.dropout,
                         kernel_initializer=glorot_uniform(seed=26),
                         recurrent_initializer=orthogonal(seed=54))

        self.tanh = Activation('tanh')
        self.softmax = Activation('softmax')

        self.W_w_dropout = Dropout(self.dropout)
        self.W_p_dropout = Dropout(self.dropout)
        self.W_s_dropout = Dropout(self.dropout)

        self.W_w = Dense(units=self.hidden_size,
                         kernel_initializer=glorot_uniform(seed=32),
                         input_shape=(self.hidden_size,))
        self.W_p = Dense(units=self.hidden_size,
                         kernel_initializer=glorot_uniform(seed=49),
                         input_shape=(2 * self.hidden_size, ))
        self.W_s = Dense(units=self.hidden_size,
                         kernel_initializer=glorot_uniform(seed=31),
                         input_shape=(2 * self.hidden_size, ))

        self.fc1_Dense = Dense(units=2 * self.hidden_size,
                               activation='relu',
                               kernel_initializer=he_normal(seed=84))
        self.fc1_dropout = tDropout(self.dropout)

        self.fc = Dense(units=self.num_classes, activation='softmax',
                        kernel_initializer=glorot_uniform(seed=91),
                        input_shape=(self.hidden_size,))

        return

    def call(self, question, image):
        # Image: B x 196 x 512 -> B x 196 x 512
        image = self.image_dense(image)
        image = self.tanh(image)

        # Words: B x L -> B x L x 512
        words = self.embed(question)

        # B x L x 512 -> B x 1 x L x 512
        unigrams = tf.expand_dims(self.tanh(self.unigram_conv(words)), 1)
        bigrams = tf.expand_dims(self.tanh(self.bigram_conv(words)), 1)
        trigrams = tf.expand_dims(self.tanh(self.trigram_conv(words)), 1)

        # phrase: # (B x 1 x L x 512, B x 1 x L x 512, B x 1 x L x 512) -> B x L x 512
        phrase = tf.squeeze(self.max_pool(
            tf.concat((unigrams, bigrams, trigrams), 1)), axis=1)
        phrase = self.tanh(phrase)
        phrase = self.phrase_dropout(phrase)

        hidden = None
        # B x L x 512 -> B x L x 512
        sentence = self.lstm(phrase)

        # B x 196 x 512, B x L x 512 -> B x 512, B x 512
        v_word, q_word = self.co_attention(image, words)

        # B x 196 x 512, B x L x 512 -> B x 512, B x 512
        v_phrase, q_phrase = self.co_attention(image, phrase)

        # B x 196 x 512, B x L x 512 -> B x 512, B x 512
        v_sent, q_sent = self.co_attention(image, sentence)

        #  B x 512, B x 512 -> B x 512
        h_w = self.tanh(self.W_w(self.W_w_dropout(q_word + v_word)))
        h_p = self.tanh(self.W_p(self.W_p_dropout(
            tf.concat(((q_phrase + v_phrase), h_w), axis=1))))
        h_s = self.tanh(self.W_s(self.W_s_dropout(
            tf.concat(((q_sent + v_sent), h_p), axis=1))))

        # B x 512 -> B X 1024
        fc1 = self.fc1_Dense(self.fc1_dropout(h_s))
        logits = self.fc(fc1)

        return logits

    def co_attention(self, img_feat, ques_feat):
        # B x 512 x 196, B x 512 x 196
        img_corr = self.image_corr(img_feat)

        # B x L x 512, B x 512 x 196 -> B x L x 196
        weight_matrix = tf.keras.backend.batch_dot(
            ques_feat, img_corr, axes=(2, 2))
        weight_matrix = self.tanh(weight_matrix)

        # B x L x 512 -> B x L x 512
        ques_embed = self.image_atten_dense(ques_feat)

        # B x 512 x 196 -> B x 196 x 512
        img_embed = self.question_atten_dens(img_feat)

        # B x L x 196, B x 196 x 512 -> B x L x 512
        transform_img = tf.keras.backend.batch_dot(weight_matrix, img_embed)

        # B x L x 512
        ques_atten_sum = self.tanh(transform_img + ques_embed)
        ques_atten_sum = self.question_atten_dropout(ques_atten_sum)
        # B x L x 512 -> B x L x 1
        ques_atten = self.ques_atten(ques_atten_sum)

        # B x L x 1 -> B x L
        ques_atten = tf.keras.layers.Reshape(
            (ques_atten.shape[1],))(ques_atten)
        ques_atten = self.softmax(ques_atten)

        # atten for image feature
        # B x L x 196, B x L x 512 -> B x 196 x 512
        transform_ques = tf.keras.backend.batch_dot(
            weight_matrix, ques_embed, axes=(1, 1))
        # B x 196 x 512,  B x 196 x 512 -> B x 196 x 512
        img_atten_sum = self.tanh(transform_ques+img_embed)
        img_atten_sum = self.image_atten_dropout(img_atten_sum)

        # B x 196 x 512 -> B x 196 x 1
        img_atten = self.img_atten(img_atten_sum)

        # B x 196 x 1 -> B x 196
        img_atten = tf.keras.layers.Reshape((img_atten.shape[1],))(img_atten)
        img_atten = self.softmax(img_atten)

        # B x L -> B x 1 x L
        ques_atten = tf.keras.layers.Reshape(
            (1, ques_atten.shape[1]))(ques_atten)
        # B x 196 -> B x 1 x 196
        img_atten = tf.keras.layers.Reshape((1, img_atten.shape[1]))(img_atten)

        # B x 1 x L, B x L x 512 -> B x 1 x 512
        ques_atten_feat = tf.keras.backend.batch_dot(ques_atten, ques_feat)
        # B x 1 x 512 -> B x 512
        ques_atten_feat = tf.keras.layers.Reshape(
            (ques_atten_feat.shape[-1],))(ques_atten_feat)

        # B x 1 x 196, B x 512 x 196 -> B x 1 x 512
        img_atten_feat = tf.keras.backend.batch_dot(img_atten, img_feat)
        # B x 1 x 512 -> B x 512
        img_atten_feat = tf.keras.layers.Reshape(
            (img_atten_feat.shape[-1],))(img_atten_feat)

        return img_atten_feat, ques_atten_feat
