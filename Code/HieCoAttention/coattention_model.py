import json
from constants import *
from coattention_layer import *
from prepare_generator import *
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping


def coattention():
    image_input = Input(shape=(196, 512))
    question_input = Input(shape=(SEQ_LENGTH,))

    output = CoattentionModel()(question_input, image_input)

    model = Model(inputs=[question_input, image_input], outputs=output)

    return model


def scheduler(epoch):
    if epoch < 10:
        return 0.0001
    else:
        return 0.0001 * tf.math.exp(0.1 * (10 - epoch))


def Train(dataset=True):

    train_generator, val_generator, val_question_ids = get_generator(dataset)

    save_config(dataset)

    checkpoint = ModelCheckpoint(CHECKPOINT_PATH + '/cp-{epoch: 04d}.ckpt',
                                 save_weights_only=True,
                                 verbose=1)

    scheduler_lr = LearningRateScheduler(scheduler, verbose=0)
    earlystop_callback = EarlyStopping(monitor='val_loss', patience=3)

    model = coattention()
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # Save the weights using the `checkpoint_path` format
    model.save_weights(CHECKPOINT_PATH +
                       '/cp-{epoch: 04d}.ckpt'.format(epoch=0))

    history = model.fit(x=train_generator,
                        epochs=EPOCHS,
                        validation_data=val_generator,
                        callbacks=[checkpoint, earlystop_callback],
                        workers=6,
                        use_multiprocessing=True)

    # save history
    with open(HISTORY_PATH, 'w') as file:
        json.dump(history.history, file)

    # prediction
    predictions = model.predict(val_generator,
                                workers=6,
                                use_multiprocessing=True,
                                verbose=1)

    ans_vocab = load_ans_vocab()

    result = []
    for q in range(len(val_question_ids)):
        ans = ans_vocab[str(predictions[q].argmax(axis=-1))]
        q_id = int(val_question_ids[q])
        result.append({u'answer': ans, u'question_id': q_id})

    with open(PRED_PATH, 'w') as file:
        json.dump(list(result), file)

    return


def save_config(dataset):
    if dataset == 0:
        DATASET = 'English'
    if dataset == 1:
        DATASET = 'Google'
    if dataset == 2:
        DATASET = 'Targoman'

    config = {'NAME': 'coattention',
              'EMBEDDING': 'keras',
              "OPTIMIZER": 'Adam',
              "LOSS": 'categorical_crossentropy',
              'DROPOUT_RATE': DROPOUT_RATE,
              "EMBEDDING_DIM": EMBEDDING_DIM,
              "EPOCHS": EPOCHS,
              "BATCH_SIZE": BATCH_SIZE,
              "SEQ_LENGTH": SEQ_LENGTH,
              "NUM_CLASSES": NUM_CLASSES, }

    print("save config in" + str(CONFIG_PATH))
    with open(CONFIG_PATH, 'w') as file:
        json.dump(list(config), file)

    return


Train(dataset=1)
