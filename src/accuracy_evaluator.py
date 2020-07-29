import csv
import cv2 as cv

from image_pre_processor import get_mouth_pixels_without_preprocessing, pre_process_image
from recurrent_predictor import get_talking


def test_frames(vid_name, labels):
    f = open("../" + vid_name + "_labelled.csv", 'w', newline='')
    with f:
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
                    # mouth_pixels = get_mouth_pixels_without_preprocessing(image)
                    # print(mouth_pixels)
                    # if mouth_pixels is not None:
                    #     pre_processed_pixels = pre_process_image(mouth_pixels)
                    # print(pre_processed_pixels)
                    # TODO ~~~~~~~~~~~~~
                    talking = get_talking(image)
                    # print(talking)
                    if frame % 100 == 0:
                        print("FRAMES: " + str(frame))
                        print("ACCURACY: " + str((correct_frames / (frame + 1)) * 100))
                        print("INCORRECTLY SUPPRESSED: " + str(
                            (incorrectly_suppressed / (open_mouth_frames + 1)) * 100))
                        print("CORRECTLY SUPPRESSED: " + str(
                            (correctly_suppressed / ((frame - open_mouth_frames) + 1)) * 100))
                        print(" -- ")

                    if curr_min <= frame <= curr_max:
                        # writer.writerow([pre_processed_pixels.tolist(), '1'])
                        # else:
                        #     writer.writerow([pre_processed_pixels.tolist(), '0'])

                        open_mouth_frames += 1
                        if talking:
                            correct_frames += 1
                        else:
                            incorrectly_suppressed += 1
                    else:
                        if not talking:
                            correct_frames += 1
                            correctly_suppressed += 1

                    if frame > curr_max:
                        open_range_no += 1
                        curr_min = int(labels[open_range_no][0])
                        curr_max = int(labels[open_range_no][1])
                        # print("     FRAMES: " + str(frame))
                        # print("     CURR MIN: " + str(curr_min))
                        # print("     CURR MAX: " + str(curr_max))

                # cv.waitKey(1)

                frame += 1
            print("ACCURACY: " + str((correct_frames / frame) * 100))
            print("INCORRECTLY SUPPRESSED: " + str((incorrectly_suppressed / frame) * 100))
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


test_accuracy()
