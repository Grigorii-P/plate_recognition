import os
import cv2
import sys
import re
from time import time
from os.path import join
from math import sqrt
from constants import *
from pytesseract import TesseractError
from Levenshtein import distance
from NomeroffNet import filters, RectDetector, TextDetector, RegionDetector, Detector, textPostprocessing


FRAMES_PER_SEC = 50
FRAME_DIFF = 30
PLATE_OFFSET = 110
TRASH_FILTER = 4  # min elems num to appear on red box
LEV_THRESHOLD = 1


def is_close(coord, center, pl_offset):
    x1, y1 = coord[0], coord[1]
    x2, y2 = center[0], center[1]
    return sqrt((x1 - x2)**2 + (y1 - y2)**2) < pl_offset


def get_center(points):
    p1 = points[0]
    p2 = points[2]
    x = (p1[0] + p2[0]) // 2
    y = (p1[1] + p2[1]) // 2
    return (x, y)


def match_plates(new_coord, coords, cur_text, target_dict, cur_frame, pl_offset):
    # if found close elems - print its box
    to_del = []
    # coords = {(x, y): [frame, target_text, color]}
    for coord in coords.keys():
        if is_close(coord, new_coord, pl_offset):
            # if we encounter a plate that disappeared quite a bit ago
            if cur_frame - coords[coord][0] > FRAME_DIFF:
                to_del.append(coord)
                continue

            text, color = coords[coord][1], coords[coord][2]
            del coords[coord]
            coords[new_coord] = [cur_frame, text, color]
            return color, text

    for coord in to_del:
        del coords[coord]

    # if coord is not close to those present on the frame,
    # and it is in target list - print its box as well

    # take the closest word to the target (if any)
    check_green_dists = [x for x in target_dict if (distance(
        cur_text, x) <= LEV_THRESHOLD and target_dict[x] == 0)]
    check_black_dists = [x for x in target_dict if (distance(
        cur_text, x) <= LEV_THRESHOLD and target_dict[x] == 1)]

    if any(check_green_dists):
        text = check_green_dists[0]
        coords[new_coord] = [cur_frame, text, 'green']
        return 'green', text

    if any(check_black_dists):
        text = check_black_dists[0]
        coords[new_coord] = [cur_frame, text, 'blue']
        return 'blue', ''

    return 'red', ''


def print_line_and_text(img, points, text, color):
    for i in range(-1, 3, 1):
        p1 = tuple([int(x) for x in list(points[i])])
        p2 = tuple([int(x) for x in list(points[i + 1])])
        cv2.line(img, p1, p2, color, LINE_THICKNESS)

    min_x, min_y = 10000, 10000
    for p in points:
        if p[0] < min_x:
            min_x = p[0]
        if p[1] < min_y:
            min_y = p[1]

    # put recognized text
    corner_of_text = (int(min_x) + SHIFT, int(min_y) - SHIFT)
    cv2.putText(img, text, corner_of_text, FONT,
                FONT_SCALE, color, LINE_TYPE)


def check_0_o(s):
    s = list(s)
    if s[0] == '0':
        s[0] = 'O'
    for i in range(1, 3, 1):
        if s[i] == 'O':
            s[i] = '0'
    for i in range(4, 6, 1):
        if s[i] == '0':
            s[i] = 'O'

    if s[-1] == 'O':
        s[-1] = '0'
    return ''.join(s)


def video_inference(src, dst, target_dict, pl_offset):
    src_vid = cv2.VideoCapture(src)
    _, img = src_vid.read()
    w_h = (img.shape[1], img.shape[0])
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    dst_vid = cv2.VideoWriter(dst, fourcc, FRAMES_PER_SEC, w_h)

    # coords = {(x, y): [frame, target_text, color]}
    # consists of plates that were found in target list at least once
    coords = {}

    frame_count = 0
    while True:
        ret, img = src_vid.read()
        frame_count += 1
        if ret is False:
            break

        img = cv2.resize(img, w_h)

        NP = nnet.detect([img])
        cv_img_masks = filters.cv_img_mask(NP)

        for img_mask in cv_img_masks:

            # detect plate and recognize text
            points = rectDetector.detect(
                img_mask, fixRectangleAngle=1, outboundWidthOffset=3)
            zone = rectDetector.get_cv_zones(img, points)
            text = re.sub('[^A-Za-z0-9]', '',
                          textDetector.detect(zone).upper())
            if len(text) >= 6:
                text = check_0_o(text)

            zone_center = get_center(points)

            # process bboxes to find matches between frames
            color, correct_text = match_plates(
                zone_center, coords, text, target_dict, frame_count, pl_offset)

            # print line and text for bboxes
            if color == 'green':
                print_line_and_text(img, points, correct_text, GREEN_COLOR)
            elif color == 'blue':
                print_line_and_text(img, points, correct_text, BLUE_COLOR)
            elif color == 'red':
                if len(text) >= TRASH_FILTER:
                    print_line_and_text(img, points, '', RED_COLOR)

        dst_vid.write(img)

        if frame_count % 100 == 0:
            print(frame_count)

        # if frame_count == 250:
        #     break

    cv2.destroyAllWindows()
    src_vid.release()
    dst_vid.release()


if __name__ == '__main__':
    NOMEROFF_NET_DIR = ''
    MASK_RCNN_DIR = '../Data/models/'
    MASK_RCNN_LOG_DIR = os.path.join(NOMEROFF_NET_DIR, '../logs/')
    MASK_RCNN_MODEL_PATH = os.path.join(
        NOMEROFF_NET_DIR, '../Data/models/mask_rcnn_numberplate_0700.h5')
    REGION_MODEL_PATH = os.path.join(
        NOMEROFF_NET_DIR, '../Data/models/imagenet_vgg16_np_region_2019_1_18.h5')
    sys.path.append(NOMEROFF_NET_DIR)

    # Initialize npdetector with default configuration file.
    nnet = Detector(MASK_RCNN_DIR, MASK_RCNN_LOG_DIR)
    # Load weights in keras format.
    nnet.loadModel(MASK_RCNN_MODEL_PATH)
    # Initialize rect detector with default configuration file.
    rectDetector = RectDetector()
    # Initialize text detector.
    textDetector = TextDetector()
    # # Initialize numberplate region detector.
    # regionDetector = RegionDetector()
    # regionDetector.load(REGION_MODEL_PATH)

    path_src = '../Data/video_results/preprocessed'
    path_dst = '../Data/video_results/result/'

    videos = {
        'IMG_7460.MOV': IMG_7460,
        # 'IMG_7461.MOV': IMG_7461,
        # 'IMG_7462.MOV': IMG_7462,
        'IMG_7463.MOV': IMG_7463,
        'IMG_7464.MOV': IMG_7464,
        'IMG_7465.MOV': IMG_7465,
        'IMG_7466.MOV': IMG_7466,
        'IMG_7467.MOV': IMG_7467,
        'IMG_7501.MOV': IMG_7501,
        'IMG_7502.MOV': IMG_7502,
        'IMG_7503.MOV': IMG_7503,
        'IMG_7504.MOV': IMG_7504,
        'IMG_7505.MOV': IMG_7505,
        'IMG_7506.MOV': IMG_7506,
    }

    # todo прогнать все видео через
    #  tessdata_best

    for video in videos:
        print(video)
        t0 = time()
        try:
            video_inference(join(path_src, video), join(
                path_dst, video), videos[video], PLATE_OFFSET)
        except TesseractError:
            os.system('rm %s/%s' % (path_dst, video))
            continue

        print('time - {} min'.format((time() - t0)/60))
