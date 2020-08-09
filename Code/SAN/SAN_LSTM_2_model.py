from constants import *
from image_layer import *
from attention_layer import *
from prepare_generator import *
from question_layer_LSTM import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD


def SAN_LSTM_2(num_classes, dropout_rate, num_words, embedding_dim, attention_dim):

    qs_input = Input(shape=(SEQ_LENGTH,))
    img_input = Input(shape=(512, 14, 14))

    image_embed = image_layer()(img_input)
    ques_embed = question_layer_LSTM(num_words,
                                     embedding_dim,
                                     dropout_rate,
                                     SEQ_LENGTH)(qs_input)

    att = attention_layer(attention_dim)([image_embed, ques_embed])
    att = attention_layer(attention_dim)([image_embed, att])

    att = Dropout(dropout_rate)(att)

    output = Dense(num_classes, activation='softmax')(att)

    model = Model(inputs=[qs_input, img_input], outputs=output)

    return model


def Train(google=True):
    """
    Train SAN_LSTM_2  with 2 attention layer.
    """

    train_generator, val_generator = get_generator(google)

    if google:
        checkpoint_path = 'checkpoint\SAN_LSTM_2_google.h5'
    else:
        checkpoint_path = 'checkpoint\SAN_LSTM_2_targoman.h5'

    checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True)

    model = SAN_LSTM_2(NUM_CLASSES,
                       DROPOUT_RATE,
                       VOCAB_SIZE,
                       EMBEDDING_DIM,
                       ATTENTION_DIM)

    optimizer = SGD(learning_rate=LR, momentum=0.9)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(x=train_generator,
                        epochs=EPOCHS,
                        validation_data=val_generator,
                        callbacks=[checkpoint])
    # save history
    return history


Train(google=True)
# Train(google=False)
