import os

BASE_PATH = '/home/ubuntu/vqa/'

# Question path (Google translation)
GOOGLE_QUESTION_TRAIN_PATH = os.path.join(
    BASE_PATH, 'dataset/google/google-train.json')
GOOGLE_QUESTION_VAL_PATH = os.path.join(
    BASE_PATH, 'dataset/google/google-val.json')

# Question path (Targoman translation)
TARGOMAN_QUESTION_TRAIN_PATH = os.path.join(
    BASE_PATH, 'dataset/targoman/targoman-train.json')
TARGOMAN_QUESTION_VAL_PATH = os.path.join(
    BASE_PATH, 'dataset/targoman/targoman-val.json')
