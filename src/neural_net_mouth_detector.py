import os

from PIL import Image
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, Conv1D, Flatten
import cv2 as cv
import numpy as np
import csv


def get_dataset():
    y = []
    f = open('../faces_mouth_open.csv', 'r')
    with f:
        reader = csv.reader(f)
        for row in reader:
            y.append(row)  # TODO: does this need to be row[0]?

    return y


def get_mouth_pixels(image):
    face_cascade_name = '../haarcascade_frontalface_alt.xml'
    face_cascade = cv.CascadeClassifier()
    # -- 1. Load the cascades
    if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)
    frame_gray = cv.cvtColor(np.float32(image), cv.COLOR_BGR2GRAY)
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


def get_data(labels, labels_camera):
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


def get_camera_dataset():
    y = []
    f = open('../camera_mouth_labels.csv', 'r')
    with f:
        reader = csv.reader(f)
        for row in reader:
            row_data = []
            for data in row:
                row_data.append(data)  # TODO: does this need to be row[0]?
            y.append(row_data)

    return y


def get_labelled_data():
    y = []
    f = open('../mouth_labelled.csv', 'r')
    with f:
        reader = csv.reader(f)
        for row in reader:
            row_data = []
            for data in row:
                row_data.append(data)  # TODO: does this need to be row[0]?
            y.append(row_data)

    return y


def define_model():
    new_model = Sequential()
    # new_model.add(Dense(128, input_dim=255
    #                     , activation='relu'))
    new_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(256, 1)))
    new_model.add(Flatten())
    # new_model.add(Dropout(0.1))
    new_model.add(Dense(64, activation='relu'))
    # new_model.add(Dropout(0.1))
    new_model.add(Dense(1, activation='sigmoid'))
    new_model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')

    return new_model


def train_model(x, y, model_inp):
    model_inp.fit(x, y, epochs=650, batch_size=64, validation_split=0.2, verbose=2)
    _, accuracy = model_inp.evaluate(x, y)
    print("Accuracy: %.2f" % (accuracy * 100))


# X_1 = get_dataset()
# X_2 = get_camera_dataset()
# get_data(X_1, X_2)


xy = get_labelled_data()
all_x = []
all_y = []

for row in xy:
    all_x.append(row[:-1])
    all_y.append([row[-1]])

# all_x = Input(shape=(490,256))
all_x = np.array(all_x).reshape((-1, 1, 255))
all_x = all_x.astype(np.float)
all_x = tf.convert_to_tensor(all_x)
print(all_x.shape)
all_y = np.array(all_y).reshape((-1, 1, 1))
all_y = all_y.astype(np.float)
all_y = tf.convert_to_tensor(all_y)
print(all_y.shape)

model = define_model()
train_model(all_x, all_y, model)
model.save('../utk_model')

