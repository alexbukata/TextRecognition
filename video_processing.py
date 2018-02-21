import os
import shutil

import cv2
import numpy as np


def extract_frames(videofile_name):
    vidcap = cv2.VideoCapture(videofile_name)

    folder_name = videofile_name[:videofile_name.rfind('.') - 1]
    if os.path.exists(folder_name) and os.path.isdir(folder_name):
        shutil.rmtree(folder_name)
    os.mkdir(folder_name)
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        cv2.imwrite((folder_name + "/frame%d.jpg") % count, image)  # save frame as JPEG file
        count += 1


def prepare_threshold(current_frame, threshold_percent):
    full = np.full(current_frame.shape, 255)
    return np.linalg.norm(full) * threshold_percent


def frame_difference(videofile_name):
    vidcap = cv2.VideoCapture(videofile_name)
    success, current_frame = vidcap.read()
    threshold = prepare_threshold(current_frame, 0.025)
    previous_frame = current_frame
    while success:
        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

        frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)
        if np.linalg.norm(frame_diff) > threshold:
            cv2.imshow('frame diff ', previous_frame)
            frame_timestamp = int(vidcap.get(cv2.CAP_PROP_POS_MSEC))
            frame_timestamp = '{}:{}'.format(int(frame_timestamp / 60000), int(frame_timestamp / 1000))
            print(frame_timestamp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        previous_frame = current_frame.copy()
        ret, current_frame = vidcap.read()
    vidcap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    frame_difference("pgj1.avi")
