import cv2.cv2
import cv2 as cv
import numpy as np

base_total_min = -1
base_template = []
suppression_delay = 0
good_match_run = 0
prev_template = []


def get_mouth_pixels_without_preprocessing(image):
    from camera_mouth_detector import face_cascade
    faces = face_cascade.detectMultiScale(image)

    # TODO: change so it doesnt just take first face?
    if len(faces) > 0:
        (x, y, w, h) = faces[0]

        center_x = x + w // 2
        start_y = y + ((2 * h) // 3)

        image = image[start_y: y+h, center_x - w // 8: center_x + w // 8]

        cv.imshow("mouth", image)
        # cv.waitKey(0)

        return image
    return None


def pre_process_image(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # image = image.convertTo(image, cv.cv2.CV_32F)
    # image = cv.cv2.UMat.get(image)
    image = np.float32(image)
    image = cv.GaussianBlur(image, (5, 5), 0)
    image = cv.reduce(image, 1, cv.REDUCE_AVG)
    image = cv.normalize(image, None, 1.0, 0.0, cv.cv2.NORM_MINMAX)

    return image


def mouth_threshold(mouth_pixels):
    last_max = -1

    thresh_series = cv.threshold(mouth_pixels, 0.3, 1, cv.THRESH_TOZERO)[1]

    # print(thresh_series)

    for x in range(len(thresh_series)):
        if thresh_series[x][0] == 0.0:
            if last_max > 0 and x - last_max < 25:
                # Join the two maximums.
                for mx in range(last_max + 1, x):
                    thresh_series[mx] = 0.0

            last_max = x

    return thresh_series


def update_base(new_base):
    global base_template, base_total_min
    min_count = 0
    # get the center section of the mouth
    base_template = new_base[16:224]
    for x in range(len(new_base)):
        if new_base[x] == 0:
            min_count += 1
    base_total_min = min_count


def process(grey):
    global suppression_delay, good_match_run, prev_template

    play = False
    series = get_mouth_pixels_without_preprocessing(grey)
    series = pre_process_image(series)
    series = cv.resize(series, (1, 256), 0, 0, cv.INTER_LINEAR)

    min_value = 0

    suppression_delay -= 1
    if suppression_delay > 0:
        play = True

    if base_total_min >= 0:
        match_results = cv.matchTemplate(series, base_template, cv.TM_SQDIFF)
        (min_value, _, _, _) = cv.minMaxLoc(match_results)

    # if min_value <= 15 or base_total_min == -1:
    min_count = 0
    thresh_series = mouth_threshold(series)

    for x in range(len(thresh_series)):
        if thresh_series[x] == 0:
            min_count += 1

    if min_value <= 15 and base_total_min >= 0:

        if min_count > base_total_min * 1.5:
            play = True
            suppression_delay = 30

    if base_total_min == -1:
        update_base(thresh_series)
    else:
        match_results = cv.matchTemplate(series, prev_template, cv.TM_SQDIFF)
        (min_value, _, _, _) = cv.minMaxLoc(match_results)

        if min_value <= 0.2:
            if good_match_run > 10:
                update_base(thresh_series)
                # cv.imshow("match", match_results)
                # cv.waitKey(0)
            else:
                good_match_run += 1
        else:
            good_match_run = 0

    prev_template = series[16:224]
    # print(suppression_delay)
    return play
