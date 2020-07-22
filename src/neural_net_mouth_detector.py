import os

import keras
from PIL import Image
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Flatten, BatchNormalization
import cv2 as cv
import numpy as np
import csv

# extract mouth pixels from image
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizers import SGD

from mouth_detector import pre_process_image

disable_eager_execution()


def get_mouth_pixels(image):
    face_cascade_name = '../haarcascade_frontalface_alt.xml'
    face_cascade = cv.CascadeClassifier()
    # -- 1. Load the cascades
    if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)
    frame_gray = cv.cvtColor(np.float32(image), cv.COLOR_BGR2GRAY)
    frame_gray = pre_process_image(frame_gray)
    frame_gray = np.uint8(frame_gray)

    faces = face_cascade.detectMultiScale(frame_gray)

    # TODO: change so it doesnt just take first face?
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        height, width = frame_gray.shape
        resize_factor = 255 / (h // 3)
        new_width = int(width * resize_factor)
        new_height = int(height * resize_factor)
        resized_frame = cv.resize(frame_gray, (new_width, new_height))

        center_x = x + w // 2
        start_y = y + ((2 * h) // 3)

        pixels = []
        for i in range(255):
            pixels.append(resized_frame[center_x, start_y + i])

        return pixels
    return None


def get_celeb_mouth_data(celeb_labels):
    celeb_data_dir = '../celeb_interviews'
    celeb_name_dirs = os.listdir(celeb_data_dir)
    curr_celeb_ind = 0
    curr_celeb_dir = celeb_name_dirs[curr_celeb_ind]

    celeb_subdirs_path = celeb_data_dir + "/" + curr_celeb_dir + "/1.6"
    celeb_subdirs = os.listdir(celeb_subdirs_path)
    curr_subdir_ind = 0
    curr_subdir = celeb_subdirs[curr_subdir_ind]
    curr_subdir_pics = os.listdir(celeb_subdirs_path + "/" + curr_subdir)

    pic_ind = 0
    no_pics = len(curr_subdir_pics)
    no_dirs = len(celeb_subdirs)
    no_celebs = len(celeb_name_dirs)

    f = open('../celeb_mouth_labelled.csv', 'w', newline='')
    with f:
        writer = csv.writer(f)
        # change directory
        for label in celeb_labels:
            if pic_ind > no_pics - 1:
                if curr_subdir_ind < no_dirs - 1:
                    curr_subdir_ind += 1

                else:
                    if curr_celeb_ind < no_celebs - 1:
                        curr_subdir_ind = 0
                        curr_celeb_ind += 1
                        curr_celeb_dir = celeb_name_dirs[curr_celeb_ind]
                        celeb_subdirs_path = celeb_data_dir + "/" + curr_celeb_dir + "/1.6"
                        celeb_subdirs = os.listdir(celeb_subdirs_path)
                        no_dirs = len(celeb_subdirs)

                    else:
                        return

                curr_subdir = celeb_subdirs[curr_subdir_ind]
                curr_subdir_pics = os.listdir(celeb_subdirs_path + "/" + curr_subdir)
                pic_ind = 0
                no_pics = len(curr_subdir_pics)

                print("LABEL: " + label[1] + "      SUBDIR: " + curr_subdir + "     CONTENTS: " + str(no_pics))
                if label[1] is "" or not curr_subdir.startswith(label[1]):
                    print("ERROR!! curr dir: " + curr_subdir + "label name: " + label[1])
                    exit(0)

            curr_pic = curr_subdir_pics[pic_ind]
            if curr_pic.endswith(".jpg"):
                # print(file_name)
                im = Image.open(celeb_subdirs_path + "/" + curr_subdir + "/" + curr_pic)
                mouth_pixels = get_mouth_pixels(im)
                if mouth_pixels is not None:
                    mouth_pixels.append(label[0])
                    writer.writerow(mouth_pixels)
                pic_ind += 1
                # print("LABEL: " + label[1] + "      SUBDIR: " + curr_subdir)


# gets image data and writes with labels to csv file
def write_mouth_data_to_csv(labels, labels_camera):
    directory = '../UTKFace'
    directory_camera = '../camera_mouth_open_closed'

    files = os.listdir(directory)
    files_camera = os.listdir(directory_camera)

    f = open('../mouth_labelled.csv', 'w', newline='')
    with f:
        writer = csv.writer(f)
        for i in range(len(labels)):
            file_name = files[i]
            if file_name.endswith(".jpg"):
                # print(file_name)
                im = Image.open(directory + "/" + file_name)
                mouth_pixels = get_mouth_pixels(im)
                if mouth_pixels is not None:
                    mouth_pixels.append(labels[i][0])
                    writer.writerow(mouth_pixels)

        for i in range(len(labels_camera)):
            file_name = files_camera[i]
            if file_name.endswith(".jpg"):
                # print(file_name)
                im = Image.open(directory_camera + "/" + file_name)
                mouth_pixels = get_mouth_pixels(im)
                if mouth_pixels is not None:
                    mouth_pixels.append(labels_camera[i][0])
                    writer.writerow(mouth_pixels)


#
def get_labels(get_camera, get_celebs):
    y = []
    if get_camera:
        f = open('../camera_mouth_labels.csv', 'r')
    elif get_celebs:
        f = open('../celeb_faces_labels.csv', 'r')
    else:
        f = open('../faces_mouth_open.csv', 'r')
    with f:
        reader = csv.reader(f)
        for row in reader:
            y.append(row)

    return y


def get_labelled_data():
    y = []
    f = open('../mouth_labelled.csv', 'r')
    with f:
        reader = csv.reader(f)
        for row in reader:
            row_data = []
            for data in row:
                row_data.append(data)
            y.append(row_data)

    f = open('../celeb_mouth_labelled.csv', 'r')
    with f:
        reader = csv.reader(f)
        for row in reader:
            row_data = []
            for data in row:
                row_data.append(data)
            y.append(row_data)

    return y


def define_model():
    new_model = Sequential()
    # new_model.add(Dense(128, input_dim=255
    #                     , activation='relu'))
    new_model.add(BatchNormalization())
    new_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(255, 1)))
    # new_model.add(Flatten())
    new_model.add(BatchNormalization())
    new_model.add(Dropout(0.1))
    new_model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(251, 1)))
    new_model.add(Flatten())
    new_model.add(BatchNormalization())
    # new_model.add(Dense(128, activation='relu'))
    new_model.add(Dropout(0.1))
    new_model.add(Dense(1, activation='sigmoid'))
    opt = SGD(lr=0.01)
    new_model.compile(loss='binary_crossentropy', optimizer=opt)

    return new_model


def train_model(x, y, model_inp):
    model_inp.fit(x, y, epochs=200, batch_size=512, validation_split=0.2, verbose=2, steps_per_epoch=5)
    _, accuracy = model_inp.evaluate(x, y)
    print("Accuracy: %.2f" % (accuracy * 100))


def get_model_data():
    xy = get_labelled_data()
    x = []
    Y = []

    for row in xy:
        x.append(row[:-1])
        Y.append([row[-1]])

    # all_x = Input(shape=(490,256))
    x = np.array(x).reshape((-1, 255, 1))
    x = x.astype(np.float)
    x = tf.convert_to_tensor(x)
    print(x.shape)
    Y = np.array(Y).reshape((-1, 1, 1))
    Y = Y.astype(np.float)
    Y = tf.convert_to_tensor(Y)
    print(Y.shape)
    return x, Y


def train_and_save_model(x, Y):
    model = define_model()
    train_model(x, Y, model)
    model.save('../utk_model.h5')


def get_data():
    y_1 = get_labels(get_camera=False)
    y_2 = get_labels(get_camera=True)
    write_mouth_data_to_csv(y_1, y_2)


# def predict(inp):
#     return model.predict(inp)


# y = get_labels(False, True)
# get_celeb_mouth_data(y)

# y_1 = get_labels(False)
# y_2 = get_labels(True)
# write_mouth_data_to_csv(y_1, y_2)
#
get_x, get_Y = get_model_data()
train_and_save_model(get_x, get_Y)

# model = load_model("../utk_model.h5")

# im = Image.open("../camera_mouth_open_closed/WIN_20200710_12_13_15_Pro.jpg")
# mouth_pixels = get_mouth_pixels(im)
# pixels = np.array(mouth_pixels).reshape((1, 255, 1))
# pixels = pixels.astype(np.float)
# pixels = tf.convert_to_tensor(pixels)
# pred = predict(pixels)
# print(pred)
