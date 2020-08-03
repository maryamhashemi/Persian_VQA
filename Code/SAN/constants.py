import os

BASE_PATH = '/home/ubuntu/vqa/'
QUESTION_TRAIN_PATH = os.path.join(BASE_PATH, '...')
ANNOTATION_TRAIN_PATH = os.path.join(BASE_PATH, '...')
IMAGE_TRAIN_PATH = os.path.join(BASE_PATH, 'dataset/train2014')

QUESTION_VAL_PATH = os.path.join(BASE_PATH, '...')
ANNOTATION_VAL_PATH = os.path.join(BASE_PATH, '...')
IMAGE_VAL_PATH = os.path.join(BASE_PATH, 'dataset/val2014')

QUESTION_TEST_PATH = os.path.join(BASE_PATH, '...')
ANNOTATION_TEST_PATH = os.path.join(BASE_PATH, '...')
IMAGE_TEST_PATH = os.path.join(BASE_PATH, 'dataset/test2015')

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
