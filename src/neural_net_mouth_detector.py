import os

from PIL import Image
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import cv2 as cv
import numpy as np


def get_dataset(data):
    x = data
    y = loadtxt('faces_mouth_open.csv', delimiter=',')
    return x, y

def define_model():
    model = Sequential()
    model.add(Dense(128, input_dim=255, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')

    return model

def train_model(x, y, model_inp):
    model_inp.fit(x, y, epochs=150, batch_size=128)
    _, accuracy = model_inp.evaluate(x, y)
    print("Accuracy: %.2f" % (accuracy * 100))


def get_mouth_pixels(image):
    face_cascade = cv.CascadeClassifier()
    frame_gray = cv.cvtColor(np.float32(image), cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    faces = face_cascade.detectMultiScale(frame_gray)

    # TODO: change so it doesnt just take first face?
    (x, y, w, h) = faces[0]
    height, width, _ = frame_gray.shape
    resize_factor = 255/h
    resized_frame = cv.resize(frame_gray, width*resize_factor, height*resize_factor)

    center_x = x + w // 2
    start_y = y + ((2 * h) // 3)

    pixels = []
    for i in range(255):
        pixels.append(resized_frame[center_x, start_y + i])

    return pixels


def get_data():
    image_pixels = []
    directory = '../mouth_open_closed'

    for file_name in os.listdir(directory):
        if file_name.endswith(".jpg"):
            im = Image.open(directory + "/" + file_name)
            image_pixels.append(get_mouth_pixels(im))

    return image_pixels


print(get_data())


