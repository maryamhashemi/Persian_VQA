from constants import *
from prepare_QA import *
from prepare_images import *
from data_generator import *

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('prepare_generator.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def get_generator(dataset):
    """
    build training and validation generator.

    Arguments:
    google -- a boolean that indicates whether use Google or Targoman translation data.
    dataset -- an int: 0 -> english, 1 -> google, 2 -> targoman


    Return:
    train_generator -- a generator of training data.
    val_generator -- a generator of validation data.

    """

    if dataset == 0:
        # english
        logger.info("use English dataset.")
        train_data = get_QA(ENGLISH_QUESTION_TRAIN_PATH,
                            ENGLISH_ANNOTATION_TRAIN_PATH)
        val_data = get_QA(ENGLISH_QUESTION_VAL_PATH,
                          ENGLISH_ANNOTATION_VAL_PATH)
    if dataset == 1:
        # Google translation
        logger.info("use Google Translation.")
        train_data = get_QA(GOOGLE_QUESTION_TRAIN_PATH,
                            GOOGLE_ANNOTATION_TRAIN_PATH)
        val_data = get_QA(GOOGLE_QUESTION_VAL_PATH,
                          GOOGLE_ANNOTATION_VAL_PATH)
    if dataset == 2:
        # Targoman translation
        logger.info("use Targoman Translation.")
        train_data = get_QA(TARGOMAN_QUESTION_TRAIN_PATH,
                            TARGOMAN_ANNOTATION_TRAIN_PATH)
        val_data = get_QA(TARGOMAN_QUESTION_VAL_PATH,
                          TARGOMAN_ANNOTATION_VAL_PATH)

    # choose top k frequent answers.
    train_answers = train_data["multiple_choice_answer"].values
    answer_frequency = get_answer_frequency(train_answers)
    k_frequent_answer = get_k_top_answers(NUM_CLASSES, answer_frequency)

    # filter data
    train_data = filter_questions(k_frequent_answer, train_data)

    # get questions
    train_questions = train_data["question"].values
    val_questions = val_data["question"].values

    # apply preprocessing on questions
    train_seqs, val_seqs = preprocess_question(train_questions, val_questions)
    logger.info("shape of train_seqs is" + str(train_seqs.shape))
    logger.info("shape of val_seqs is" + str(val_seqs.shape))

    # get iamge ids
    train_image_ids = train_data["image_id"].values
    val_image_ids = val_data["image_id"].values
    logger.info("shape of train_image_ids is" + str(train_image_ids.shape))
    logger.info("shape of val_image_ids is" + str(val_image_ids.shape))

    # get image feature paths
    train_image_path = get_train_img_feature_paths()
    val_image_path = get_val_img_feature_paths()
    logger.info("shape of train_image_path is " + str(len(train_image_path)))
    logger.info("shape of val_image_path is " + str(len(val_image_path)))

    # get answers
    _, train_answers, val_answers = get_train_val_label(
        train_data, val_data, dataset)

    train_generator = DataGenerator(train_seqs,
                                    train_image_ids,
                                    train_image_path,
                                    train_answers,
                                    BATCH_SIZE)
    logger.info("successfully build train generator")

    val_generator = DataGenerator(val_seqs,
                                  val_image_ids,
                                  val_image_path,
                                  val_answers,
                                  BATCH_SIZE)
    logger.info("successfully build val generator")

    return train_generator, val_generator
