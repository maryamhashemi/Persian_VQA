import json
import hazm
import heapq
import pickle
import logging
import numpy as np
import pandas as pd
import deepdish as dd
from constants import *
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('prepare_QA.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def merge_QA(questions, annotations):
    """
    merge questions and annotations.

    Arguments:
    questions -- a dictionary that consists of image_id, question and question_id.
    annotations -- a dictionary that consists of answer_type, answers, image_id,
                   multiple_choice_answer, question_id and question_type.

    Return:
    QA -- a dataframe with 7 columns that are image_id, question, question_id, answer_type, answers,
            multiple_choice_answer and question_type.

    """
    questions = pd.DataFrame(questions)
    annotations = pd.DataFrame(annotations)

    QA = pd.merge(questions, annotations,  how='inner', left_on=[
        'image_id', 'question_id'], right_on=['image_id', 'question_id'])

    logger.info("successfully merge questions and answers.")
    return QA


def shuffle_data(data):
    """
    shuffle the input data.

    Arguments:
    data -- a dataframe that consists of questions and answers.

    Return:
    data -- shuffled version of the input data

    """
    data = data.sample(frac=1, random_state=123)
    logger.info("successfully shuffle data.")
    return data


def get_QA(questions_path, annotations_path):
    """
    load question and annotation files and merge their contents and return the merged data as a dataframe.

    Arguments:
    questions_path -- a string that shows questions file path.
    annotations_path -- a string that shows the annotations file path.

    Return:
    QA -- a dataframe that consists of questions and answers.

    """
    questions = json.load(open(questions_path, encoding='utf-8'))
    questions = questions["questions"]
    logger.info("successfully load questions.")

    annotations = json.load(open(annotations_path, encoding='utf-8'))
    annotations = annotations["annotations"]
    logger.info("successfully load annotations.")

    QA = merge_QA(questions, annotations)
    QA = shuffle_data(QA)
    logger.info("total number of data is %i ." % (len(QA)))
    return QA


def get_answer_frequency(answers):
    """
    calculate the frequency of each answer and return a dictionary of all answer frequencies.

    Arguments:
    answers -- a string vector that consists of answers.

    Return:
    answer_frequency -- a dictionary that maps answer to frequency.

    """
    answer_frequency = {}

    for answer in answers:
        if(answer_frequency.get(answer, -1) > 0):
            answer_frequency[answer] += 1
        else:
            answer_frequency[answer] = 1

    logger.info("successfully calculate all answer frequencies.")
    return answer_frequency


def get_k_top_answers(k, answer_frequency):
    """
    choose the top k most frequent answers.

    Arguments:
    k -- an integer that indicates the number of answers(classes).
    answer_frequency -- a dictionary that maps answer to frequency.

    Return:
    k_frequent_answers -- a dictionary that indicates top k most frequent answers.

    """
    k_frequent_answers = heapq.nlargest(k,
                                        answer_frequency,
                                        key=answer_frequency.get)

    logger.info("successfully choose the top %i most frequent answers." % (k))
    return k_frequent_answers


def filter_questions(k_frequent_answers, data):
    """
    remove the questions that their answers don't exist in top k frequent answers.

    Arguments:
    k_frequent_answers -- a dictionary that indicates top k most frequent answers.
    data -- a dataframe that consists of questions and answers.

    Return:
    filtered_data -- a dataframe that filtered by answers.

    """
    data['multiple_choice_answer'] = data['multiple_choice_answer'].apply(
        lambda x: x if x in k_frequent_answers else '')

    filtered_data = data[data['multiple_choice_answer'].apply(
        lambda x:len(x) > 0)]

    logger.info("successfully remove the questions from data.")
    return filtered_data


def get_train_val_label(train_data, val_data):
    """
    convert answers of training and validation questions to one hot vector.

    Arguments:
    train_data -- a dataframe that consists of training questions and answers.
    val_data -- a dataframe that consists of validation questions and answers.

    Return:
    ans_vocab -- a dictionary that maps answers to an int number(label).
    train_Y -- a numpy array that has shape of (number of training example, NUM_CLASSES).
    val_Y -- a numpy array that has shape of (number of validation example, NUM_CLASSES).

    """
    label_encoder = LabelBinarizer()

    train_Y = label_encoder.fit_transform(
        train_data['multiple_choice_answer'].apply(lambda x: x).values)
    logger.info("successfully get train labels.")

    val_Y = label_encoder.transform(
        val_data['multiple_choice_answer'].apply(lambda x: x).values)
    logger.info("successfully get val labels.")

    ans_vocab = {i: l for i, l in enumerate(label_encoder.classes_)}

    with open(ANS_VOCAB_PATH, 'w') as file:
        json.dump(ans_vocab, file)

    logger.info("ans_vocab dictionary saved in " + ANS_VOCAB_PATH)
    logger.info("Number of clasess: " + str(len(ans_vocab)))
    logger.info("Shape of Answer Vectors in training Data: " +
                str(train_Y.shape))
    logger.info("Shape of Answer Vectors in validation Data: " +
                str(val_Y.shape))

    return ans_vocab, train_Y, val_Y


def preprocess_question(train_qs, val_qs):
    """
    tokenize questions. Convert the tokenized questions into sequences and then pad sequences.

    Arguments:
    train_qs -- a list of training questions.
    val_qs -- a list of validatin questions.

    Return:
    train_X_seqs -- a numpy array that has shape of (number of training example, SEQ_LENGTH).
    val_X_seqs --  a numpy array that has shape of (number of validation example, SEQ_LENGTH).

    """
    # normalize data
    normalizer = hazm.Normalizer()

    train_qs = [normalizer.normalize(item) for item in train_qs]
    val_qs = [normalizer.normalize(item) for item in val_qs]

    train_qs = [item.replace('؟', ' ؟') for item in train_qs]
    val_qs = [item.replace('؟', ' ؟') for item in val_qs]

    tokenizer = Tokenizer(oov_token=OOV_TOK)
    tokenizer.fit_on_texts(train_qs)
    word_index = tokenizer.word_index
    VOCAB_SIZE = len(word_index) + 1
    logger.info("number of unique words in train is " + str(VOCAB_SIZE))

    with open(TOKEN_PATH, 'wb') as file:
        pickle.dump(tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("tokenizer saved in " + str(TOKEN_PATH))

    # prepare train sequence
    train_X_seqs = tokenizer.texts_to_sequences(train_qs)
    train_X_seqs = pad_sequences(
        train_X_seqs, maxlen=SEQ_LENGTH, padding='pre')
    train_X_seqs = np.array(train_X_seqs)
    logger.info("convert train questions to sequences.")

    # prepare val sequence
    val_X_seqs = tokenizer.texts_to_sequences(val_qs)
    val_X_seqs = pad_sequences(val_X_seqs, maxlen=SEQ_LENGTH, padding='pre')
    val_X_seqs = np.array(val_X_seqs)
    logger.info("convert validation questions to sequences.")

    vocab_and_vectors = load_fasttext()
    embedding_matrix = get_embedding_matrix(
        VOCAB_SIZE, word_index, vocab_and_vectors)

    logger.info("shape of embedding_matrix is " + str(embedding_matrix.shape))

    return train_X_seqs, val_X_seqs, embedding_matrix


def load_fasttext():
    """
    load fasttext embedding file.
    """
    file = open(FASTTEXT_PATH, encoding="utf8")

    vocab_and_vectors = {}
    for line in file:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        vocab_and_vectors[word] = vector

    logger.info("load fasttext.")

    return vocab_and_vectors


def get_embedding_matrix(vocab_size, word_index, vocab_and_vectors):
    """
    create embedding matrix of all vocabs.

    Arguments:
    word_index -- a dictionary that maps word to index.
    vocab_and_vectors -- a dictionary that maps fasttext words to their embedding vectors.

    Return:
    embedding_matrix -- a numpy array that hold words embedding.
    """
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

    for word, i in word_index.items():
        embedding_vector = vocab_and_vectors.get(word)

        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    logger.info("create embedding matrix.")

    return embedding_matrix


def load_ans_vocab():
    with open(ANS_VOCAB_PATH, 'r') as file:
        return json.load(file)
