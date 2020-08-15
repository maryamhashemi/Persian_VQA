import os

BASE_PATH = '/home/ubuntu/vqa/'

# Image path
IMAGE_TRAIN_PATH = os.path.join(BASE_PATH, 'dataset/train2014')
IMAGE_VAL_PATH = os.path.join(BASE_PATH, 'dataset/val2014')
IMAGE_TEST_PATH = os.path.join(BASE_PATH, 'dataset/test2015')

# Image feature path
IMG_FEATURE_TRAIN_PATH = os.path.join(BASE_PATH, 'dataset/vgg16/train')
IMG_FEATURE_VAL_PATH = os.path.join(BASE_PATH, 'dataset/vgg16/val')
IMG_FEATURE_TEST_PATH = os.path.join(BASE_PATH, 'dataset/vgg16/test')

# Question path (English)
ENGLISH_QUESTION_TRAIN_PATH = os.path.join(
    BASE_PATH, 'dataset/english/OpenEnded_mscoco_train2014_questions.json')
ENGLISH_QUESTION_VAL_PATH = os.path.join(
    BASE_PATH, 'dataset/english/OpenEnded_mscoco_val2014_questions.json')

# Annotation path (English)
ENGLISH_ANNOTATION_TRAIN_PATH = os.path.join(
    BASE_PATH, 'dataset/english/mscoco_train2014_annotations.json')
ENGLISH_ANNOTATION_VAL_PATH = os.path.join(
    BASE_PATH, 'dataset/english/mscoco_val2014_annotations.json')


# Question path (Google translation)
GOOGLE_QUESTION_TRAIN_PATH = os.path.join(
    BASE_PATH, 'dataset/google/google-train.json')
GOOGLE_QUESTION_VAL_PATH = os.path.join(
    BASE_PATH, 'dataset/google/google-val.json')
GOOGLE_QUESTION_TEST_PATH = os.path.join(
    BASE_PATH, 'dataset/google/google-test.json')

# Annotation path (Google translation)
GOOGLE_ANNOTATION_TRAIN_PATH = os.path.join(
    BASE_PATH, 'dataset/english/mscoco_train2014_annotations.json')
GOOGLE_ANNOTATION_VAL_PATH = os.path.join(
    BASE_PATH, 'dataset/english/mscoco_val2014_annotations.json')

# Question path (Targoman translation)
TARGOMAN_QUESTION_TRAIN_PATH = os.path.join(
    BASE_PATH, 'dataset/targoman/targoman-train.json')
TARGOMAN_QUESTION_VAL_PATH = os.path.join(
    BASE_PATH, 'dataset/targoman/targoman-val.json')
TARGOMAN_QUESTION_TEST_PATH = os.path.join(
    BASE_PATH, 'dataset/targoman/targoman-test.json')

# Annotation path (Targoman translation)
TARGOMAN_ANNOTATION_TRAIN_PATH = os.path.join(
    BASE_PATH, 'dataset/english/mscoco_train2014_annotations.json')
TARGOMAN_ANNOTATION_VAL_PATH = os.path.join(
    BASE_PATH, 'dataset/english/mscoco_val2014_annotations.json')


# Hyperparametrs
DROPOUT_RATE = 0.5
EMBEDDING_DIM = 512
EPOCHS = 50
BATCH_SIZE = 300
SEQ_LENGTH = 26
VOCAB_SIZE = 0
NUM_CLASSES = 1000
LR = 0.0005
# LR = 1

OOV_TOK = "<OOV>"

# Experiment id
EXP_ID = 2

# Tokenizer path
TOKEN_PATH = 'Exp{id}/tokenizer.pickle'.format(id=EXP_ID)

# answer to label dictionary path
ANS_VOCAB_PATH = 'Exp{id}/ans_vocab.json'.format(id=EXP_ID)

# fasttext embedding path
FASTTEXT_PATH = os.path.join(BASE_PATH, 'dataset/cc.fa.300.vec')

# checkpoints path
CHECKPOINT_PATH = 'Exp{id}/checkpoint'.format(id=EXP_ID)

# history path
HISTORY_PATH = 'Exp{id}/history.json'.format(id=EXP_ID)

# predictions path
PRED_PATH = 'Exp{id}/result.json'.format(id=EXP_ID)

# config path
CONFIG_PATH = 'Exp{id}/config.json'.format(id=EXP_ID)
