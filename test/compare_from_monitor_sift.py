import pyautogui
import cv2
import numpy as np
import time

from core.sift_compare import sift_compare
from core.sys_info import get_screen_resolutions

template = cv2.imread('../img/exit.png', cv2.IMREAD_GRAYSCALE)
screens = get_screen_resolutions()
main_screen = next((screen for screen in screens if screen['is_main']), None)
main_screen_width = 0
main_screen_height = 0
if main_screen:
    main_screen_width = main_screen['width']
    main_screen_height = main_screen['height']
while True:
    screen = np.array(pyautogui.screenshot())
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    result = sift_compare(template, screen_gray)
    if 0 < result[0] <= main_screen_width and 0 < result[1] <= main_screen_height and result[4] <= main_screen_width and \
            result[5] <= main_screen_height:
        print(result[4], result[5])
        break
    time.sleep(1)
