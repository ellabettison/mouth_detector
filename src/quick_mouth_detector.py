from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np

long_norm = None
norm = None
last_five_frames = np.array([0, 0, 0])


def detectAndDisplay(frame):
    global norm, long_norm, last_five_frames
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    # -- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)

    new_frame = None
    last_five_frames_mean = 0
    for (x, y, w, h) in faces:
        frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        print(x, y, w, h)
        frame = cv.rectangle(frame, (x + w // 2, y + (2 * h // 3)), (x + w // 2, y + h), (255, 0, 0), 3)

        center_x = x + w // 2
        start_y = y + ((2 * h) // 3)

        rgb_pixels = []
        # for rgb in range(3):
        for i in range(h // 3):
            rgb_pixels.append(frame_gray[center_x, start_y + i])

        new_frame = np.array(rgb_pixels, dtype=np.float64)

        # new_frame -= new_frame.min()
        # # Normalise
        new_frame *= 255 / new_frame.max(initial=1)

        # new_frame = (new_frame - new_frame.mean())/new_frame.std()

        # new_frame = (new_frame**2 / np.linalg.norm(new_frame))*10
        for i in range(h // 3):
            frame = cv.circle(frame, (center_x, start_y + i), 1, (new_frame[i], new_frame[i], new_frame[i]), 2)

        # print(new_frame.mean())
        # print(new_frame.max())
        # print(new_frame.min())

        threshold = (new_frame.mean() - new_frame.min()) / 2

        min_threshold = new_frame.mean() - new_frame.std()
        max_threshold = new_frame.mean() + new_frame.std()

        curr_dark_frames = 0
        dark_frames = 0
        # for rgb in range(3):
        for i in range(len(new_frame)):
            # print(new_frame[i], min_threshold)
            if new_frame[i] < min_threshold:
                curr_dark_frames += 1
            else:
                if curr_dark_frames > dark_frames:
                    dark_frames = curr_dark_frames
                curr_dark_frames = 0

        if long_norm is None:
            long_norm = dark_frames
        else:
            if (dark_frames - last_five_frames_mean) > 2 or (last_five_frames_mean - dark_frames) > 2:
                long_norm = (9 * long_norm + last_five_frames_mean) // 10

        if norm is None:
            norm = dark_frames
        else:
            last_five_frames = last_five_frames[1:]
            last_five_frames = np.append(last_five_frames, [norm])
            # print(last_five_frames)
            last_five_frames_mean = last_five_frames.mean()

            if abs(last_five_frames_mean - long_norm) > 3:
                frame = cv.rectangle(frame, (0, 0), (50, 50), (255, 0, 0), 5)
            norm = (4 * norm + dark_frames) // 5
            frame = cv.putText(frame, str(last_five_frames_mean), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
                               cv.LINE_AA)

        frame = cv.putText(frame, str(long_norm), (200, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv.LINE_AA)

    cv.imshow('Capture - Face detection', frame)
    return new_frame


np.set_printoptions(precision=3)

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.',
                    default='haarcascade_frontalface_alt.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)

args = parser.parse_args()
face_cascade_name = args.face_cascade
face_cascade = cv.CascadeClassifier()
# -- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
camera_device = args.camera
# -- 2. Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    if cv.waitKey(10) == 27:
        break
