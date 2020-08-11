import json
from constants import *
from coattention_layer import *
from prepare_generator import *
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint


def coattention():
    image_input = Input(shape=(196, 512))
    question_input = Input(shape=(SEQ_LENGTH,))
    output = CoattentionModel()(image_input, question_input)

    model = Model(inputs=[question_input, image_input], outputs=output)

    return model


def scheduler(epoch):
    if epoch < 10:
        return 0.0001
    else:
        return 0.0001 * tf.math.exp(0.1 * (10 - epoch))


def Train(google=True):

    train_generator, val_generator = get_generator(google)

    if google:
        checkpoint_path = 'checkpoint/coattention_google/cp-{epoch:04d}.ckpt'
        history_path = 'trainHistoryDict/coattention_google.json'
    else:
        checkpoint_path = 'checkpoint/coattention_targoman/cp-{epoch:04d}.ckpt'
        history_path = 'trainHistoryDict/coattention_targoman.json'

    checkpoint = ModelCheckpoint(checkpoint_path,
                                 save_weights_only=True,
                                 verbose=1)
    scheduler_lr = LearningRateScheduler(scheduler, verbose=0)

    model = coattention()
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # Save the weights using the `checkpoint_path` format
    model.save_weights(checkpoint_path.format(epoch=0))

    history = model.fit(x=train_generator,
                        epochs=EPOCHS,
                        validation_data=val_generator,
                        callbacks=[checkpoint, scheduler_lr])
    # save history
    with open(history_path, 'w') as file:
        json.dump(history.history, file)

    return


Train(google=True)
# Train(google=False)
