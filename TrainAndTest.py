import cv2
import numpy as np
from keras.optimizers import SGD
from keras.models import load_model
import CustomZeroPadding

import DataProcessing
import NetworkModel

if __name__ == '__main__':
    # load data
    train_labels_map = DataProcessing.read_labels()
    train_images, train_labels = DataProcessing.read_images(train_labels_map)
    train_images = DataProcessing.preprocess_images(train_images)

    # build model
    max_height = 60
    max_width = 100
    input_shape = (1, max_height, max_width)
    max_length = 33
    alphabet_length = 37

    # train network
    network_train_output = DataProcessing.words_to_matrix(np.array(train_labels), max_length, alphabet_length)
    network_train_images = train_images.reshape(train_images.shape[0], 1, max_height, max_width)
    want_load = False
    if want_load:
        model = load_model("D:\\my_model.hdf5")
    else:
        model = NetworkModel.build_model(input_shape, max_length, alphabet_length)
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)
        model.fit(network_train_images, network_train_output, batch_size=4, epochs=2, verbose=1)
        model.save("D:\\my_model.hdf5")

    predict_result = model.predict(network_train_images[0].reshape(1, 1, 60, 100))
    print(predict_result)
    answer_r = predict_result.reshape(33, 37)
    chars = []
    output_str = ' 0123456789abcdefghijklmnopqrstuvwxyz'  # space was the last
    output = [x for x in output_str]
    for row in answer_r:
        chars.append(output[np.argmax(row)])
    print(''.join(chars))
    print("Reading done")
    print(train_labels[0])
    cv2.imshow("1", train_images[4])
    cv2.waitKey(0)
