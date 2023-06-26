import cv2


def read_img(path, grayscale=True):
    # 读取模板图像和全屏截图
    if isinstance(path, str):
        if grayscale:
            path = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            path = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return path
