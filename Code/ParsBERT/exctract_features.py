import os
import re
import json
import logging
import numpy as np
import pandas as pd
from constants import *
from transformers import pipeline
from transformers import AutoConfig, AutoTokenizer, TFAutoModel

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('extract_features.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def load_questions(questions_path):
    """
    read the path of questions in 'questions_path' and return a dataframe consists of questions.

    Arguments:
    questions_path -- a directory that consists of questions.

    Return:
    questions -- a dataframe that consists of questions.

    """
    questions = json.load(open(questions_path, encoding='utf-8'))
    questions = questions["questions"]
    logger.info("successfully load questions.")
    questions = pd.DataFrame(questions)

    return questions


def extract_features(questions, questions_id, dir):
    """
    extract features from questions using ParsBert.

    Arguments:
    questions -- a list that consists of questions.
    questions_id -- a list that consists of questions_id.
    dir -- a path for saving features.

    """
    config = AutoConfig.from_pretrained(
        "HooshvareLab/bert-base-parsbert-uncased")
    tokenizer = AutoTokenizer.from_pretrained(
        "HooshvareLab/bert-base-parsbert-uncased")
    model = TFAutoModel.from_pretrained(
        "HooshvareLab/bert-base-parsbert-uncased")
    pipe = pipeline('feature-extraction', model=model, tokenizer=tokenizer)

    questions = [q.replace('؟', ' ؟') for q in questions]

    num_ques = len(questions)

    for i, (q, q_id) in enumerate(zip(questions, questions_id)):
        features = pipe(q)
        features = np.squeeze(features)
        np.save(BASE_PATH + dir + str(q_id) + '.npy', features)

        if (i+1) % 100 == 0:
            logger.info("extract features from %i/%i questions." %
                        (i + 1, num_ques))

    return


def save_features(questions_path, save_dir):
    """
    extract features from train questions using ParsBERT and save them as .npy file.

    """
    questions = load_questions(questions_path)

    ques = questions["question"].values
    ques_id = questions["question_id"].values

    extract_features(ques, ques_id, save_dir)

    return


logger.info("extracting features from google train")
save_features(GOOGLE_QUESTION_TRAIN_PATH, 'dataset/parsbert/google/train/')

# logger.info("extracting features from targoman train")
# save_features(TARGOMAN_QUESTION_TRAIN_PATH, 'dataset/parsbert/targoman/train/')

# logger.info("extracting features from goolge val")
# save_features(GOOGLE_QUESTION_VAL_PATH, 'dataset/parsbert/google/val/')

# logger.info("extracting features from targoman val")
# save_features(TARGOMAN_QUESTION_VAL_PATH, 'dataset/parsbert/targoman/val/')
