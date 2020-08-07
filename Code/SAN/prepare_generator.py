from data_generator import *
from constants import *
from prepare_QA import *
from prepare_images import *

train_data = get_QA(GOOGLE_QUESTION_TRAIN_PATH, GOOGLE_ANNOTATION_TRAIN_PATH)
val_data = get_QA(GOOGLE_QUESTION_VAL_PATH, GOOGLE_ANNOTATION_VAL_PATH)
test_data = get_QA(GOOGLE_QUESTION_TEST_PATH, GOOGLE_ANNOTATION_TEST_PATH)

# get questions
train_questions = train_data["question"].values
val_questions = val_data["question"].values
test_question = test_data["question"].values

# apply preprocessing on questions
train_seqs, val_seqs, test_seqs = preprocess_question(
    train_questions, val_questions, test_question)

# get iamge ids
train_image_ids = train_data["image_id"].values
val_image_ids = val_image_ids["image_id"].values

# get image feature paths
train_image_path = get_train_img_feature_paths()
val_image_path = get_val_img_feature_paths()

# get answers
train_answers = None  # not yet implemented
val_answers = None  # not yet implemented


train_generator = DataGenerator(train_seqs,
                                train_image_ids,
                                train_image_path,
                                train_answers,
                                BATCH_SIZE,
                                NUM_CLASSES)

val_generator = DataGenerator(val_seqs,
                              val_image_ids,
                              val_image_path,
                              val_answers,
                              BATCH_SIZE,
                              NUM_CLASSES)
