# -*- coding: utf-8 -*-
"""探针：验证「窗口级捕获」能否在窗口被遮挡时拿到画面。

用法：
  python window_capture.py --list                # 列出所有可见窗口
  python window_capture.py --test <窗口标题关键字>  # 尝试捕获并保存 png，输出可用率
"""
import os
import sys

import numpy as np
import cv2

try:
    import win32gui
    import win32ui
    import win32con
    PYWIN32_OK = True
except Exception:
    PYWIN32_OK = False

PW_RENDERFULLCONTENT = 0x00000002


def list_windows():
    """列出有标题、未隐藏的顶层窗口：[(hwnd, 标题)]"""
    result = []

    def cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
            result.append((hwnd, win32gui.GetWindowText(hwnd)))
        return True

    win32gui.EnumWindows(cb, None)
    return result


def capture_window(hwnd):
    """用 PrintWindow(PW_RENDERFULLCONTENT) 抓取窗口画面，返回 BGR ndarray 或 None"""
    if not PYWIN32_OK:
        return None
    try:
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        w, h = right - left, bottom - top
        if w <= 0 or h <= 0:
            return None

        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
        save_dc.SelectObject(bitmap)

        # PW_RENDERFULLCONTENT：请求窗口离屏重绘，部分 DX/OpenGL 程序也支持
        ok = ctypes_print_window(hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT)
        if not ok:
            ctypes_print_window(hwnd, save_dc.GetSafeHdc(), 0)

        bits = bitmap.GetBitmapBits(True)
        img = np.frombuffer(bits, dtype=np.uint8).reshape(h, w, 4)
        img = img[:, :, :3]  # BGRA -> BGR
        return img
    except Exception as exc:
        print('capture_window 失败: %s' % exc)
        return None
    finally:
        try:
            win32gui.ReleaseDC(hwnd, hwnd_dc)
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.DeleteObject(bitmap.GetHandle())
        except Exception:
            pass


def ctypes_print_window(hwnd, hdc, flags):
    import ctypes
    return ctypes.windll.user32.PrintWindow(hwnd, hdc, flags)


def usable_ratio(img):
    """非纯黑像素占比，用来判断抓到的是不是有效画面"""
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float((gray > 12).mean())


def main():
    if '--list' in sys.argv:
        for hwnd, title in list_windows()[:40]:
            print('%s\t%s' % (hwnd, title[:60]))
        return 0

    if '--hwnd' in sys.argv:
        hwnd = int(sys.argv[sys.argv.index('--hwnd') + 1])
        img = capture_window(hwnd)
        if img is None:
            print('捕获失败')
            return 1
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'win_capture_%s.png' % hwnd)
        cv2.imencode('.png', img)[1].tofile(path)
        print('hwnd=%s size=%dx%d 有效像素=%.1f%% -> %s' % (
            hwnd, img.shape[1], img.shape[0], usable_ratio(img) * 100, path))
        return 0

    if '--test' in sys.argv:
        key = sys.argv[sys.argv.index('--test') + 1]
        hits = [(h, t) for h, t in list_windows() if key.lower() in t.lower()]
        if not hits:
            print('未找到包含 %s 的窗口' % key)
            return 1
        out = []
        for hwnd, title in hits[:3]:
            img = capture_window(hwnd)
            ratio = usable_ratio(img)
            if img is not None:
                path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'win_capture_%s.png' % hwnd)
                cv2.imencode('.png', img)[1].tofile(path)
                out.append('hwnd=%s title=%s 有效像素=%.1f%% -> %s' % (hwnd, title[:30], ratio * 100, path))
            else:
                out.append('hwnd=%s title=%s 捕获失败' % (hwnd, title[:30]))
        print('\n'.join(out))
        return 0 if any('有效像素' in o and '有效像素=0.0' not in o for o in out) else 1

    print(__doc__)
    return 0


if __name__ == '__main__':
    sys.exit(main())
