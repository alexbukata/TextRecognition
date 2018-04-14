import csv

import cv2
import numpy as np
from numpy import mean, std
from skimage.transform import resize

output_str = ' 0123456789abcdefghijklmnopqrstuvwxyz'  # space was the last
output = [x for x in output_str]


def read_labels():
    train_labels_path = "D:\\text_recognition_data\\COCO-Text-words-trainval\\train_words_gt_upgrade.txt"
    # train_data = map(lambda x: train_data_path+"\\"+x, os.listdir(train_data_path))
    # train_data = list(map(cv2.imread, train_data))
    with(open(train_labels_path, 'r', encoding='UTF-8')) as train_labels_file:
        reader = csv.reader(train_labels_file, delimiter='~')
        reader = [a for index, a in enumerate(reader) if index < 5000]
        train_labels = dict(reader)
    return train_labels


def read_images(train_labels):
    train_data_path = 'D:\\text_recognition_data\\COCO-Text-words-trainval\\train_words'
    X = []
    y = []
    for filename, value in train_labels.items():
        X.append(cv2.imread("{}\\{}.jpg".format(train_data_path, filename)))
        filtered_word = ''.join(map(lambda char: char if char in output else ' ', value.lower()))
        y.append(filtered_word)
    return X, y


def preprocess_images(images):
    result = []
    for index, image in enumerate(images):
        # old_height, old_width = image.shape[:2]
        new_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        new_im = resize(new_im, (60, 100), order=1, preserve_range=True, mode='constant')
        new_im = (new_im - mean(new_im)) / (std(new_im) + 0.0001)
        # new_im = cv2.copyMakeBorder(new_im, 0, height_max - old_height, 0, width_max - old_width, cv2.BORDER_CONSTANT, value=color)
        result.append(new_im)
    return np.array(result)


def words_to_matrix(words, max_length=33, max_alphabet=37):
    result = np.zeros((words.shape[0], max_length, max_alphabet))
    for k, word in enumerate(words):
        word_ = word.lower()
        word_matrix = np.zeros((max_length, max_alphabet))
        ind_row = 0
        for char in word_:
            if char in output:
                word_matrix[ind_row][output.index(char)] = 1
            ind_row += 1
        result[k] = word_matrix
    return result.reshape(result.shape[0], max_length * max_alphabet)
