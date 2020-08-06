import os

BASE_PATH = '/home/ubuntu/vqa/'

# Image path
IMAGE_TRAIN_PATH = os.path.join(BASE_PATH, 'dataset/train2014')
IMAGE_VAL_PATH = os.path.join(BASE_PATH, 'dataset/val2014')
IMAGE_TEST_PATH = os.path.join(BASE_PATH, 'dataset/test2015')

# Question path (Google translation)
GOOGLE_QUESTION_TRAIN_PATH = os.path.join(
    BASE_PATH, 'dataset/google/google-train.json')
GOOGLE_QUESTION_VAL_PATH = os.path.join(
    BASE_PATH, 'dataset/google/google-val.json')
GOOGLE_QUESTION_TEST_PATH = os.path.join(
    BASE_PATH, 'dataset/google/google-test.json')

# Annotation path (Google translation)
GOOGLE_ANNOTATION_TRAIN_PATH = os.path.join(
    BASE_PATH, 'dataset/google/google-train-annotations.json')
GOOGLE_ANNOTATION_VAL_PATH = os.path.join(
    BASE_PATH, 'dataset/google/google-val-annotations.json')
GOOGLE_ANNOTATION_TEST_PATH = os.path.join(BASE_PATH, '...')

# Question path (Targoman translation)
TARGOMAN_QUESTION_TRAIN_PATH = os.path.join(
    BASE_PATH, 'dataset/google/targoman-train.json')
TARGOMAN_QUESTION_VAL_PATH = os.path.join(
    BASE_PATH, 'dataset/google/targoman-val.json')
TARGOMAN_QUESTION_TEST_PATH = os.path.join(
    BASE_PATH, 'dataset/google/targoman-test.json')

# Annotation path (Targoman translation)
TARGOMAN_ANNOTATION_TRAIN_PATH = os.path.join(
    BASE_PATH, 'dataset/google/targoman-train-annotations.json')
TARGOMAN_ANNOTATION_VAL_PATH = os.path.join(
    BASE_PATH, 'dataset/google/targoman-val-annotations.json')
TARGOMAN_ANNOTATION_TEST_PATH = os.path.join(BASE_PATH, '...')

# Hyperparametrs
DROPOUT_RATE = 0.5
EMBEDDING_DIM = 100
EPOCHS = 20
BATCH_SIZE = 64
SEQ_LENGTH = 40
VOCAB_SIZE = 1000
NUM_FILTERS = [256, 256, 512]
FILTER_SIZE = [1, 2, 3]
ATTENTION_DIM = 301
OOV_TOK = "<OOV>"
