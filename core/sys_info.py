import win32api
import win32con


def get_screen_resolutions():
    screens = []
    dev_num = 0
    while True:
        device = win32api.EnumDisplayDevices(None, dev_num, 1)
        if not device.StateFlags & win32con.DISPLAY_DEVICE_ATTACHED_TO_DESKTOP:
            break
        dm = win32api.EnumDisplaySettings(device.DeviceName, -1)
        width = dm.PelsWidth
        height = dm.PelsHeight
        is_main = True if device.StateFlags & win32con.DISPLAY_DEVICE_PRIMARY_DEVICE else False
        screens.append({'width': width, 'height': height, 'is_main': is_main})
        dev_num += 1
    return screens
