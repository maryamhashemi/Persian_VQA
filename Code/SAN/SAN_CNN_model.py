from constants import *
from image_layer import *
from question_layer_CNN import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import, Dense, Input


def SAN_CNN(num_classes, dropout_rate, num_words, embedding_dim, attention_dim):

    qs_input = Input(shape=(SEQ_LENGTH,))
    img_input = Input(shape=(512, 14, 14))

    image_embed = image_layer()(img_input)
    ques_embed = question_layer_CNN(num_words,
                                    embedding_dim,
                                    FILTER_SIZE,
                                    NUM_FILTERS,
                                    SEQ_LENGTH,
                                    dropout_rate)(qs_input)

    att = attention_layer(attention_dim)([image_embed, ques_embed])
    att = attention_layer(attention_dim)([image_embed, att])

    output = Dense(num_classes, activation='softmax')(att)

    model = Model(inputs=[qs_input, img_input], outputs=output)
    return model


def Train():
    checkpoint = ModelCheckpoint('model_SAN_LSTM_2.h5', save_best_only=True)

    model = SAN_CNN(num_classes, DROPOUT_RATE, VOCAB_SIZE,
                    EMBEDDING_DIM, ATTENTION_DIM)

    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    history = model.fit([train_X_seqs, train_X_ims],
                        train_Y,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=([val_X_seqs, val_X_ims], val_Y),
                        callbacks=[checkpoint])
    return history
