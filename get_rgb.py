import cv2
import numpy as np

def get_rgb(img_path):
    #for resizing
    target_W = 100
    target_H = 100

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

    # 시간 오래 걸리지 않게, 800*800 to 100*100
    # 근데 화질이 너무 깨짐.
    # img = cv2.resize(img, (target_width,target_height))

    img = img.astype(np.float32) / 255.0

    #RGBA to RGB
    if img.shape[2] == 4:
        alpha = img[..., 3:]
        rgb = img[..., :3]

        img = rgb * alpha + (1.0 - alpha)

    return img