import json
from constants import *
from image_layer import *
from attention_layer import *
from prepare_generator import *
from question_layer_LSTM import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers.schedules import ExponentialDecay


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
        checkpoint_path = 'checkpoint/SAN_LSTM_2_google/cp-{epoch:04d}.ckpt'
        history_path = 'trainHistoryDict/SAN_LSTM_2_google.json'
    else:
        checkpoint_path = 'checkpoint/SAN_LSTM_2_targoman/cp-{epoch:04d}.ckpt'
        history_path = 'trainHistoryDict/SAN_LSTM_2_targoman.json'

    checkpoint = ModelCheckpoint(checkpoint_path,
                                 save_weights_only=True,
                                 verbose=1)

    model = SAN_LSTM_2(NUM_CLASSES,
                       DROPOUT_RATE,
                       VOCAB_SIZE,
                       EMBEDDING_DIM,
                       ATTENTION_DIM)

    lr_schedule = ExponentialDecay(initial_learning_rate=LR,
                                   decay_steps=10000,
                                   decay_rate=0.99997592083)

    optimizer = Adam(learning_rate=lr_schedule, clipnorm=10)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # Save the weights using the `checkpoint_path` format
    model.save_weights(checkpoint_path.format(epoch=0))

    history = model.fit(x=train_generator,
                        epochs=EPOCHS,
                        validation_data=val_generator,
                        callbacks=[checkpoint])

    # save history
    with open(history_path, 'w') as file:
        json.dump(history.history, file)

    return history


Train(google=True)
# Train(google=False)
