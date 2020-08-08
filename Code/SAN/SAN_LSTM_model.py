from constants import *
from image_layer import *
from attention_layer import *
from prepare_generator import *
from question_layer_LSTM import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint


def SAN_LSTM(num_classes, dropout_rate, num_words, embedding_dim, attention_dim):

    qs_input = Input(shape=(SEQ_LENGTH,))
    img_input = Input(shape=(512, 14, 14))

    image_embed = image_layer()(img_input)
    ques_embed = question_layer_LSTM(num_words,
                                     embedding_dim,
                                     dropout_rate,
                                     SEQ_LENGTH)(qs_input)

    att = attention_layer(attention_dim)([image_embed, ques_embed])
    att = attention_layer(attention_dim)([image_embed, att])

    output = Dense(num_classes, activation='softmax')(att)

    model = Model(inputs=[qs_input, img_input], outputs=output)

    return model


def Train():

    checkpoint = ModelCheckpoint('/checkpoint', save_best_only=True)

    model = SAN_LSTM(NUM_CLASSES,
                     DROPOUT_RATE,
                     VOCAB_SIZE,
                     EMBEDDING_DIM,
                     ATTENTION_DIM)

    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    train_generator, val_generator = get_generator()

    history = model.fit(x=train_generator,
                        epochs=EPOCHS,
                        validation_data=val_generator,
                        callbacks=[checkpoint])

    return history


Train()
