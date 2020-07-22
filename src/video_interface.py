import cv2 as cv
from moviepy.editor import *

from mouth_detector import process


def test_frames(vid_path):
    video_capture = cv.VideoCapture(vid_path)
    fps = video_capture.get(cv.CAP_PROP_FPS)
    img_success, image = video_capture.read()

    while img_success:
        img_success, image = video_capture.read()

        if (img_success):
            play = process(image)

            cv.imshow("img", image)
            cv.waitKey(0)


test_frames('../call-16.mp4')
