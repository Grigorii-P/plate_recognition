import cv2
import os
from os.path import join
from scipy import ndimage


frames_per_sec = 60
angle = -90
meta = {
    'IMG_7460.MOV': 1,
    'IMG_7461.MOV': 1,
    'IMG_7462.MOV': 1,
    'IMG_7463.MOV': 1,
    'IMG_7464.MOV': 1,
    'IMG_7465.MOV': 1,
    'IMG_7466.MOV': 0,
    'IMG_7467.MOV': 0,
    'IMG_7501.MOV': 1,
    'IMG_7502.MOV': 1,
    'IMG_7503.MOV': 0,
    'IMG_7504.MOV': 0,
    'IMG_7505.MOV': 0,
    'IMG_7506.MOV': 1,
}


def process_video(src, dst, coef):
    cap = cv2.VideoCapture(src)
    _, img = cap.read()
    w_h_new = (int(img.shape[1] / coef), int(img.shape[0] / coef))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(dst, fourcc, frames_per_sec, w_h_new)

    frame_count = 0
    while True:
        ret, img = cap.read()
        if ret is False:
            break

        # if meta[src.split('/')[-1]]:
        #     img = ndimage.rotate(img, angle)
        # img = cv2.resize(img, w_h_new)
        video.write(img)
        cv2.imwrite('img.jpg', img)

        frame_count += 1
        if frame_count % 100 == 0:
            print(frame_count)

        # if frame_count == 10:
        #     break

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':

    path_src = 'video_results/result/final'
    path_dst = 'video_results/result/temp'

    videos = [x for x in os.listdir(path_src) if x.endswith('1.mov')]
    coefs = [1]
    for coef in coefs:
        os.system('mkdir %s/%s' % (path_dst, str(coef)))
        for v in videos:
            print(v)
            process_video(join(path_src, v), join(
                path_dst, str(coef) + '/' + v), coef)
