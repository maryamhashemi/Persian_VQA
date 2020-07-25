import os
import re
import logging
import numpy as np
import pandas as pd
from PIL import Image
from constants import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('prepare_images.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def read_image_paths(dir_path):
    """
    read the path of images in 'dir_path' and return a dictionary mapping image_id to image path.

    Arguments:
    dir_path -- a directory that consists of images.

    Return:
    ims -- a dictionary that maps image_id to image path.

    """
    ims = {}

    for filename in os.listdir(dir_path):
        if filename.endswith('.jpg'):
            image_id = int(re.findall('\d+', filename)[1])
            ims[image_id] = os.path.join(dir_path, filename)

    return ims


def load_and_proccess_image(image_path, model, image_size):
    """
    load and preprocess image, then extract features using model.

    Arguments:
    image_path -- a string that is image path.
    model -- a cnn model for extracting features from image.
    image_size -- expected size of image after loading.

    Return:
    features -- extracted features from image.

    """
    im = load_img(image_path, target_size=image_size)
    x = img_to_array(im)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)

    return features[0]


def extract_features(paths):
    """
    extract features from images using VGG19.

    Arguments:
    paths -- a dictionary that maps image_ids to image paths.

    Return:
    ims -- a dictionary that saves mapping between image_ids and image features.

    """
    ims = {}
    base_model = VGG19(weights='imagenet', include_top=True)
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('fc2').output)

    for i, (image_id, image_path) in enumerate(paths.items()):
        ims[image_id] = load_and_proccess_image(
            image_path, model, (224, 224, 3))

        if i % 1 == 100:
            logger.info("extract features form %i of images" % i)
    return ims


def get_train_image_paths():
    return read_image_paths(IMAGE_TRAIN_PATH)


def get_val_image_paths():
    return read_image_paths(IMAGE_VAL_PATH)


def get_test_image_paths():
    return read_image_paths(IMAGE_TEST_PATH)


def save_train_features():
    """
    extract features from train images using VGG19 and save them as .csv file.

    """
    logger.info("Start: extract features from train images.")
    train_ims = extract_features(get_train_image_paths())
    logger.info("End: extract features from train images.")

    df = pd.DataFrame(train_ims)
    df.to_csv('X_train_ims_VGG19.csv')
    logger.info('saved in \"X_train_ims_VGG19.csv\"')


def save_val_features():
    """
    extract features from validation images using VGG19 and save them as .csv file.

    """
    logger.info("Start: extract features from val images.")
    val_ims = extract_features(get_val_image_paths())
    logger.info("End: extract features from val images.")

    df = pd.DataFrame(val_ims)
    df.to_csv('X_val_ims_VGG19.csv')
    logger.info('saved in \"X_val_ims_VGG19.csv\"')


def save_test_features():
    """
    extract features from test images using VGG19 and save them as .csv file.

    """
    logger.info("Start: extract features from test images.")
    test_ims = extract_features(get_test_image_paths())
    logger.info("End: extract features from test images.")

    df = pd.DataFrame(test_ims)
    df.to_csv('X_test_ims_VGG19.csv')
    logger.info('saved in \"X_test_ims_VGG19.csv\"')


save_train_features()
save_val_features()
save_test_features()
