from data_generator import *
from constants import *
from prepare_QA import *
from prepare_images import *

train_data = get_QA(GOOGLE_QUESTION_TRAIN_PATH, GOOGLE_ANNOTATION_TRAIN_PATH)
val_data = get_QA(GOOGLE_QUESTION_VAL_PATH, GOOGLE_ANNOTATION_VAL_PATH)
test_data = get_QA(GOOGLE_QUESTION_TEST_PATH, GOOGLE_ANNOTATION_TEST_PATH)

train_questions = train_data["question"]
val_questions = val_data["question"]
test_question = test_data["question"]

train_seqs, val_seqs, test_seqs = preprocess_question(
    train_questions, val_questions, test_question)

train_image_ids = train_data["image_id"]
val_image_ids = val_image_ids["image_id"]

train_image_path = get_train_img_feature_paths()
val_image_path = get_val_img_feature_paths()

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
