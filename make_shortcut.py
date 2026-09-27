# -*- coding: utf-8 -*-
"""在桌面生成「屏幕标识计数器」快捷方式（PowerShell 的 COM 被本机策略拦截，改用 Python COM）。"""

import os
import sys
import time

import pythoncom
import win32com.client

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(BASE_DIR, 'screen_counter.py')
VENV_DIR = os.path.dirname(os.path.dirname(sys.executable))  # .../envs/sc-counter
PYTHONW = os.path.join(os.path.dirname(sys.executable), 'pythonw.exe')
PYTHON = os.path.join(os.path.dirname(sys.executable), 'python.exe')


def get_desktop():
    """优先取当前用户的真实桌面目录（兼容 OneDrive 重定向）"""
    import winreg
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                             r'Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders')
        val, _ = winreg.QueryValueEx(key, 'Desktop')
        val = os.path.expandvars(val)
        if os.path.isdir(val):
            return val
    except Exception:
        pass
    return os.path.join(os.path.expanduser('~'), 'Desktop')


def create_shortcut(lnk_path, target, args, workdir, icon, desc):
    shell = win32com.client.Dispatch('WScript.Shell')
    sc = shell.CreateShortCut(lnk_path)
    sc.TargetPath = target
    sc.Arguments = args
    sc.WorkingDirectory = workdir
    sc.IconLocation = icon
    sc.Description = desc
    sc.save()


def read_shortcut(lnk_path):
    shell = win32com.client.Dispatch('WScript.Shell')
    sc = shell.CreateShortCut(lnk_path)
    return sc.TargetPath, sc.Arguments, sc.WorkingDirectory, sc.IconLocation


def main():
    if not os.path.exists(SCRIPT):
        print('脚本不存在: %s' % SCRIPT)
        return 1

    # pythonw.exe 启动无黑框控制台；缺失时退回 python.exe
    target = PYTHONW if os.path.exists(PYTHONW) else PYTHON
    if not os.path.exists(target):
        print('未找到解释器: %s' % target)
        return 1

    desktop = get_desktop()
    lnk = os.path.join(desktop, '屏幕标识计数器.lnk')
    create_shortcut(lnk, target, '"%s"' % SCRIPT, BASE_DIR, '%s,0' % target,
                    '屏幕区域标识自动计数工具')

    # 回读校验
    t, a, w, i = read_shortcut(lnk)
    ok = os.path.exists(lnk) and t.lower() == target.lower() and SCRIPT in a and os.path.isdir(w)
    print('快捷方式: %s' % lnk)
    print('目标:     %s' % t)
    print('参数:     %s' % a)
    print('起始位置: %s' % w)
    print('图标:     %s' % i)
    print('SHORTCUT %s' % ('OK' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    pythoncom.CoInitialize()
    sys.exit(main())
