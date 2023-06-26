import cv2
import numpy as np

from core.read_img import read_img


def sift_compare(fragment_img_path, entire_img_path, threshold=0.6, grayscale=True):
    """
    :param fragment_img_path: 待检测的碎片图片路径
    :param entire_img_path: 待检测的背景图片路径
    :param threshold: 容差，数值越小越精确，范围 0~1.0
    :param grayscale: 是否以灰度模式读取图像，默认为True
    :return: 匹配到的最小外接矩形的左上角坐标x、y、宽度、高度及中心坐标cx、cy
    """
    # 读取模板图像和全屏截图
    template = read_img(fragment_img_path, grayscale)
    img = read_img(entire_img_path, grayscale)
    # 将模板图像调整为与全屏截图相同的大小
    template = cv2.resize(template, (img.shape[1], img.shape[0]))
    # 提取特征点
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(img, None)
    # 特征点匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # 最佳匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)
    # 计算变换矩阵和匹配的角点
    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = template.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        # 绘制匹配的角点和中心点
        img = cv2.polylines(img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        x, y, w, h = cv2.boundingRect(np.int32(dst))
        cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)
        cx = x + w // 2
        cy = y + h // 2
        cv2.circle(img, (cx, cy), 5, 255, -1)
        return x, y, w, h, cx, cy
    else:
        return 0, 0, 0, 0, 0, 0
