import json
import tensorflow as tf
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


def Train(dataset):
    """
    Train SAN_LSTM_2  with 2 attention layer.

    Arguments:
    dataset -- an int: 0 -> english, 1 -> google, 2 -> targoman

    """

    train_generator, val_generator = get_generator(google)

    if dataset == 0:
        checkpoint_path = 'checkpoint/SAN_LSTM_2_english/cp-{epoch:04d}.ckpt'
        history_path = 'trainHistoryDict/SAN_LSTM_2_english.json'
    if dataset == 1:
        checkpoint_path = 'checkpoint/SAN_LSTM_2_english/cp-{epoch:04d}.ckpt'
        history_path = 'trainHistoryDict/SAN_LSTM_2_english.json'
    if dataset == 2:
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


# Train(google=True)
# Train(google=False)


def predict(dataset):

    if dataset == 0:
        checkpoint_dir = 'checkpoint/SAN_LSTM_2_englisht'
    if dataset == 1:
        checkpoint_dir = 'checkpoint/SAN_LSTM_2_english'
    if dataset == 2:
        checkpoint_dir = 'checkpoint/SAN_LSTM_2_targoman'

    # Create a new model instance
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

    # Load the previously saved weights
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)

    # predict
    loss, acc = model.predict()


def evaluate(dataset):

    if dataset == 0:
        annotations = json.load(
            open(ENGLISH_ANNOTATION_VAL_PATH, encoding='utf-8'))
    if dataset == 1:
        annotations = json.load(
            open(GOOGLE_ANNOTATION_VAL_PATH, encoding='utf-8'))
    if dataset == 2:
        annotations = json.load(
            open(TARGOMAN_ANNOTATION_VAL_PATH, encoding='utf-8'))

    answers = annotations["annotations"]
    answers_df = pd.DataFrame(answers)
    print(answers_df["answers"].values.shape)


evaluate(0)
evaluate(1)
evaluate(2)
