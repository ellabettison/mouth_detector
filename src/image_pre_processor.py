import argparse
import csv

import cv2 as cv
import numpy as np

from mouth_detector import process

template = []


def get_mouth_pixels_without_preprocessing(image):
    faces = face_cascade.detectMultiScale(image)

    # TODO: change so it doesnt just take first face?
    if len(faces) > 0:
        (x, y, w, h) = faces[0]

        center_x = x + w // 2
        start_y = y + ((2 * h) // 3)

        image = image[start_y: y + h, center_x - w // 8: center_x + w // 8]

        # cv.imshow("mouth", image)
        # cv.waitKey(0)

        return image
    return None


def pre_process_image(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = np.float32(image)
    image = cv.GaussianBlur(image, (5, 5), 0)
    image = cv.reduce(image, 1, cv.REDUCE_AVG)
    image = cv.normalize(image, None, 1.0, 0.0, cv.cv2.NORM_MINMAX)

    # TODO: NEW
    image = cv.resize(image, (1, 256), 0, 0, cv.INTER_LINEAR)

    return image


def test_frames(vid_name, labels):
    f = open("../" + vid_name + "_labelled.csv", 'w', newline='')
    with f:
        writer = csv.writer(f)
        vid_path = '../meeting_vids_done/' + vid_name + '.mp4'
        video_capture = cv.VideoCapture(vid_path)
        img_success, image = video_capture.read()
        frame = 0
        open_range_no = 0
        if labels[0][0] != '':
            curr_max = int(labels[0][1])
            curr_min = int(labels[0][0])
            correct_frames = 0
            incorrectly_suppressed = 0
            correctly_suppressed = 0
            open_mouth_frames = 0
            while img_success:
                img_success, image = video_capture.read()

                if img_success:
                    mouth_pixels = get_mouth_pixels_without_preprocessing(image)
                    # print(mouth_pixels)
                    if mouth_pixels is not None:
                        pre_processed_pixels = pre_process_image(mouth_pixels)
                        # print(pre_processed_pixels)
                        # TODO ~~~~~~~~~~~~~
                        # play = process(pre_processed_pixels)
                        if frame % 100 == 0:
                            print("FRAMES: " + str(frame))
                        #     print("ACCURACY: " + str((correct_frames / (frame+1)) * 100))
                        #     print("INCORRECTLY SUPPRESSED: " + str((incorrectly_suppressed / (open_mouth_frames+1)) * 100))
                        #     print("CORRECTLY SUPPRESSED: " + str((correctly_suppressed / ((frame - open_mouth_frames) + 1)) * 100))
                        #     print(" -- ")

                        if curr_min <= frame <= curr_max:
                            writer.writerow([pre_processed_pixels.tolist(), '1'])
                        else:
                            writer.writerow([pre_processed_pixels.tolist(), '0'])


                        #     open_mouth_frames += 1
                        #     if play:
                        #         correct_frames += 1
                        #     else:
                        #         incorrectly_suppressed += 1
                        # else:
                        #     if not play:
                        #         correct_frames += 1
                        #         correctly_suppressed += 1
                        if frame > curr_max:
                            open_range_no += 1
                            curr_min = int(labels[open_range_no][0])
                            curr_max = int(labels[open_range_no][1])
                            print("     FRAMES: " + str(frame))
                            print("     CURR MIN: " + str(curr_min))
                            print("     CURR MAX: " + str(curr_max))

                        # TODO ~~~~~~~~~~~~~
                        # if play:
                        #     image = cv.rectangle(image, (0, 0), (50, 50), (255, 0, 0), 5)
                        # print(pre_processed_pixels)

                    # cv.waitKey(1)
                    else:
                        writer.writerow([np.zeros((256, 1), dtype=float).tolist(), '1'])

                frame += 1
            # print("ACCURACY: " + str((correct_frames / frame) * 100))
            # print("INCORRECTLY SUPPRESSED: " + str((incorrectly_suppressed / frame) * 100))
            print("FINISHED " + vid_name + " \n ~~~~~~ \n")


def get_meeting_csv_labels():
    y = []
    f = open('../meeting_labels.csv', 'r')
    with f:
        curr_call = []
        reader = csv.reader(f)
        for row in reader:
            if '' != row[0]:
                y.append(curr_call)
                curr_call = [row[0]]
            curr_call.append(row[1:])
    print(y[1:])
    return y[1:]


def test_accuracy():
    y = get_meeting_csv_labels()
    for call in y:
        test_frames(call[0], call[1:])


parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.',
                    default='../haarcascade_frontalface_alt.xml')

args = parser.parse_args()
face_cascade_name = args.face_cascade
face_cascade = cv.CascadeClassifier()
# -- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)

# test_frames('../meeting_vids_done/call-2.mp4')
# test_accuracy()
