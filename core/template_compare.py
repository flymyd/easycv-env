import cv2
from core.read_img import read_img


def template_compare(fragment_img_path, entire_img_path, threshold=0.9, grayscale=True):
    template = read_img(fragment_img_path, grayscale)
    img = read_img(entire_img_path, grayscale)
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    w, h = template.shape[::-1]
    if max_val > threshold:
        x = max_loc[0]
        y = max_loc[1]
        cx = x + w // 2
        cy = y + h // 2
        return x, y, w, h, cx, cy
    else:
        return 0, 0, 0, 0, 0, 0
