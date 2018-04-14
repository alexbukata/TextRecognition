import itertools

import cv2
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.models import load_model

output_str = '0123456789abcdefghijklmnopqrstuvwxyz '  # space was the last
output = [x for x in output_str]

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 64))
    img = img.astype(np.float32)
    img /= 255
    result = np.zeros((1, 64, 128))
    result[0, :, :] = img
    return np.expand_dims(result.T, 0)


def decode_word(out):
    out_best = list(np.argmax(out[2:], 1))
    out_best = [k for k, g in itertools.groupby(out_best)]
    outstr = ''
    for c in out_best:
        if c < len(output):
            outstr += output[c]
    return outstr


if __name__ == '__main__':
    sess = tf.Session()
    K.set_session(sess)

    model = load_model("./model/awesome_model.hdf")

    net_inp = model.get_layer(name='the_input').input
    net_out = model.get_layer(name='softmax').output

    # image = cv2.imread("./images/Chevron.jpg")
    image = cv2.imread("./images/461_austins_4907.jpg")
    processed_image = preprocess(image)

    net_out_value = sess.run(net_out, feed_dict={net_inp: processed_image})

    print(decode_word(net_out_value[0]))
