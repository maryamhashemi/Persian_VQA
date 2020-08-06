from constants import *
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
                   multiple_choice_answer, question_id and question_type

    Return:
    QA -- a dataframe with 7 columns that are image_id, question, question_id, answer_type, answers, 
            multiple_choice_answer and question_type

    """
    questions = pd.DataFrame(questions)
    annotations = pd.DataFrame(annotations)

    QA = pd.merge(questions, annotations,  how='inner', left_on=[
        'image_id', 'question_id'], right_on=['image_id', 'question_id'])

    logger.info("merge questions and answers.")
    return QA


def get_QA(questions_path, annotations_path):
    """
    load question and annotation files and merge their contents and return the merged data as a dataframe

    Arguments:
    questions_path -- a string that shows questions file path.
    annotations_path -- a string that shows the annotations file path. 

    Return:
    QA -- a dataframe that consists of questions and answers.

    """
    questions = json.load(open(questions_path, encoding='utf-8'))
    questions = questions["questions"]
    logger.info("load questions.")

    annotations = json.load(open(annotations_path, encoding='utf-8'))
    annotations = annotations["annotations"]
    logger.info("load annotations.")

    QA = merge_QA(questions, annotations)

    return QA


def preprocess_question(train_qs, val_qs, test_qs):
    """
    tokenize questions. Convert the tokenized questions into sequences and then pad sequences.

    Arguments:
    train_qs -- a list of training questions.
    val_qs -- a list of validatin questions. 
    test_qs -- a list of testing questions.

    Return:
    train_X_seqs -- a numpy array that has shape of (number of training example, SEQ_LENGTH).
    val_X_seqs --  a numpy array that has shape of (number of validation example, SEQ_LENGTH).
    test_X_seqs --  a numpy array that has shape of (number of testing example, SEQ_LENGTH).
    """

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOK)
    word_index = tokenizer.word_index
    tokenizer.fit_on_texts(train_qs)

    # prepare train sequence
    train_X_seqs = tokenizer.texts_to_sequences(train_qs)
    train_X_seqs = pad_sequences(
        train_X_seqs, maxlen=SEQ_LENGTH, padding='post')
    train_X_seqs = np.array(train_X_seqs)
    logger.info("convert train questions to sequences.")

    # prepare val sequence
    val_X_seqs = tokenizer.texts_to_sequences(val_qs)
    val_X_seqs = pad_sequences(val_X_seqs, maxlen=SEQ_LENGTH, padding='post')
    val_X_seqs = np.array(val_X_seqs)
    logger.info("convert validation questions to sequences.")

    # prepare test sequence
    test_X_seqs = tokenizer.texts_to_sequences(test_qs)
    test_X_seqs = pad_sequences(test_X_seqs, maxlen=SEQ_LENGTH, padding='post')
    test_X_seqs = np.array(test_X_seqs)
    logger.info("convert test questions to sequences.")

    return train_X_seqs, val_X_seqs, test_X_seqs
