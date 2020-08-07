from constants import *
from prepare_QA import *
from prepare_images import *
from data_generator import *


def get_generator():

    # Google translation
    train_data = get_QA(GOOGLE_QUESTION_TRAIN_PATH,
                        GOOGLE_ANNOTATION_TRAIN_PATH)
    val_data = get_QA(GOOGLE_QUESTION_VAL_PATH, GOOGLE_ANNOTATION_VAL_PATH)

    # choose top k frequent answers.
    train_answers = train_data["multiple_choice_answer"].values
    answer_frequency = get_answer_frequency(train_answers)
    k_frequent_answer = get_k_top_answers(k=NUM_CLASSES, answer_frequency)

    # filter data
    train_data = filter_questions(k_frequent_answer, train_data)
    val_data = filter_questions(k_frequent_answer, val_data)

    # get questions
    train_questions = train_data["question"].values
    val_questions = val_data["question"].values

    # apply preprocessing on questions
    train_seqs, val_seqs = preprocess_question(train_questions, val_questions)

    # get iamge ids
    train_image_ids = train_data["image_id"].values
    val_image_ids = val_data["image_id"].values

    # get image feature paths
    train_image_path = get_train_img_feature_paths()
    val_image_path = get_val_img_feature_paths()

    # get answers
    _, train_answers, val_answers = get_train_val_label(train_data, val_data)

    train_generator = DataGenerator(train_seqs,
                                    train_image_ids,
                                    train_image_path,
                                    train_answers,
                                    BATCH_SIZE)

    val_generator = DataGenerator(val_seqs,
                                  val_image_ids,
                                  val_image_path,
                                  val_answers,
                                  BATCH_SIZE)

    return train_generator, val_generator
