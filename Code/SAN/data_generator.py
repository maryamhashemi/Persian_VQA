import numpy as np
from constants import *
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical


class DataGenerator(Sequence):

    def __init__(self, questions, image_ids, image_path, answers, batch_size, n_classes, shuffle=True):

        self.questions = questions
        self.image_ids = image_ids
        self.image_path = image_path
        self.answers = answers
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch

        """
        return int(np.ceil(len(self.questions) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data

        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        [x_seqs, x_ims], y = self.__data_generation(indexes)

        return [x_seqs, x_ims], y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch'
        """

        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """
        Generates data containing batch_size samples
        """

        x_seqs = np.empty((self.batch_size, SEQ_LENGTH))
        x_ims = np.empty((512, 14, 14))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i in indexes:
            # Store sample
            x_seqs[i] = self.questions[i]
            x_ims[i] = np.load(self.image_path[self.image_ids[i]])

            # Store class
            y[i] = self.answers[i]

        return [x_seqs, x_ims], to_categorical(y, num_classes=self.n_classes)
