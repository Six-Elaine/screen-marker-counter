# -*- coding: utf-8 -*-
"""
屏幕区域标识自动计数器
======================
监测屏幕某一区域，当出现指定标识（模板图片）时自动计数。

用法:
    python screen_counter.py             # 打开图形界面
    python screen_counter.py --selftest  # 无界面自检

步骤:
    1. 框选监测区域（拖拽）
    2. 截取标识模板（拖拽，或从图片加载）
    3. 开始监测 —— 标识出现 -> 计数 +1，写入 counter_log.csv
"""

import os
import sys
import csv
import json
import time
import ctypes
import threading
import datetime

import numpy as np
import cv2
import mss
import tkinter as tk
from tkinter import messagebox, filedialog

# DPI 感知：保证框选坐标 = 物理像素（必须在创建 Tk 之前）
if sys.platform == 'win32':
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

try:
    import winsound
except ImportError:
    winsound = None

try:
    from PIL import Image, ImageTk
    PIL_OK = True
except Exception:
    PIL_OK = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'config.json')
TEMPLATE_PATH = os.path.join(BASE_DIR, 'template.png')
LOG_PATH = os.path.join(BASE_DIR, 'counter_log.csv')

_MUTEX = None  # 单实例互斥量句柄，需保持引用不被 GC
WINDOW_TITLE = '屏幕区域标识自动计数器'

# 模板来源：region=在监测区域内截取（同源同尺度，最稳）
#           screen=整屏任意位置截取（灵活，需自行确保尺度一致）
TPL_SRC_REGION = '监测区域内（推荐）'
TPL_SRC_SCREEN = '整屏任意位置'

DEFAULTS = {
    'region': None,
    'threshold': 0.80,
    'mode': 'appear',      # appear=出现即计数  multi=统计可见数量
    'sound': True,
    'multiscale': False,
    'hotkeys': True,
    'interval_ms': 100,
    'count': 0,
    'max_n': 0,
    'topmost': True,
    'tpl_src': TPL_SRC_REGION,
}


def _grab_region_with(sct, region):
    """用给定 mss 实例截取 [left, top, w, h] 区域, 返回 BGR ndarray"""
    left, top, w, h = region
    raw = sct.grab({'left': int(left), 'top': int(top),
                    'width': int(w), 'height': int(h)})
    return cv2.cvtColor(np.asarray(raw), cv2.COLOR_BGRA2BGR)


def _grab_region(region):
    with mss.MSS() as sct:
        return _grab_region_with(sct, region)


class Detector:
    """灰度模板匹配 + 迟滞状态机（防止同一标识反复计数）"""

    def __init__(self, tpl_gray, threshold=0.8, multiscale=False,
                 confirm=2, miss_frames=3, cooldown=1.0):
        self.set_template(tpl_gray)
        self.multiscale = multiscale
        self.confirm = confirm          # 连续 N 帧高于阈值才确认出现
        self.miss_frames = miss_frames  # 连续 N 帧低于阈值才确认消失
        self.cooldown = cooldown        # 两次计数最小间隔（秒）
        self.set_threshold(threshold)
        self.visible = False
        self.hit = 0
        self.miss = 0
        self.last_count = 0.0

    def set_template(self, tpl_gray):
        self.tpl = tpl_gray
        self.th, self.tw = tpl_gray.shape[:2]

    def set_threshold(self, t):
        self.on_t = float(t)
        self.off_t = max(0.05, float(t) - 0.08)

    def _scales(self):
        return (0.9, 1.0, 1.1) if self.multiscale else (1.0,)

    def match_best(self, gray):
        """单实例匹配, 返回 (最高分, 左上角(x,y)或None, 缩放)"""
        best = (0.0, None, 1.0)
        gh, gw = gray.shape[:2]
        for s in self._scales():
            w = max(2, int(round(self.tw * s)))
            h = max(2, int(round(self.th * s)))
            if w > gw or h > gh:
                continue
            t = self.tpl if s == 1.0 else cv2.resize(
                self.tpl, (w, h), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, ml = cv2.minMaxLoc(res)
            if mx > best[0]:
                best = (float(mx), (int(ml[0]), int(ml[1])), s)
        return best

    def match_multi(self, gray, max_peaks=50):
        """多实例匹配（仅原始比例）, 返回 [(x, y, score), ...]"""
        th_, tw_ = self.tpl.shape[:2]
        gh, gw = gray.shape[:2]
        if th_ > gh or tw_ > gw:
            return []
        res = cv2.matchTemplate(gray, self.tpl, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= self.on_t)
        if len(xs) == 0:
            return []
        scores = res[ys, xs]
        order = np.argsort(scores)[::-1][:3000]
        r2 = (min(tw_, th_) * 0.5) ** 2
        peaks = []
        for i in order:
            x, y = int(xs[i]), int(ys[i])
            if all((x - px) ** 2 + (y - py) ** 2 > r2 for px, py, _ in peaks):
                peaks.append((x, y, float(scores[i])))
                if len(peaks) >= max_peaks:
                    break
        return peaks

    def update(self, score):
        """喂入相似度, 返回 'count' 表示该标识出现了一次（上升沿）"""
        if score >= self.on_t:
            self.hit += 1
            self.miss = 0
        elif score <= self.off_t:
            self.hit = 0
            self.miss += 1
        else:
            self.miss += 1  # 灰区: 视为衰减, 防止长期卡在"可见"
        ev = None
        if not self.visible and self.hit >= self.confirm:
            self.visible = True
            if time.time() - self.last_count >= self.cooldown:
                self.last_count = time.time()
                ev = 'count'
        elif self.visible and self.miss >= self.miss_frames:
            self.visible = False
        return ev


class _POINT(ctypes.Structure):
    """GetCursorPos 用的坐标结构（避免依赖 ctypes.wintypes）"""
    _fields_ = [('x', ctypes.c_long), ('y', ctypes.c_long)]


class SnapshotSelector:
    """冻结帧选区器：所有框选/调整都在静态快照上完成，交互对齐系统截图工具。

    坐标口径：内部一律用「快照像素坐标」，调用方再自行加上快照原点得到屏幕坐标。
    run() 返回 (x, y, w, h)；取消返回 None。
    """

    HANDLE = 8        # 手柄命中半径（显示像素）
    MIN_SIZE = 8      # 最小选区（快照像素）

    def __init__(self, root, image_bgr, title, hint, max_scale=4.0):
        self.root = root
        self.image = image_bgr
        self.ih, self.iw = image_bgr.shape[:2]
        self.title = title
        self.hint = hint
        self.max_scale = max_scale
        self.result = None
        self.sel = None       # (x0, y0, x1, y1) 快照像素
        self._drag = None
        self._photo = None

    # ---------------- 对外 ----------------
    def run(self):
        if not PIL_OK:
            messagebox.showwarning('缺少依赖', '预览需要 Pillow，请先安装 Pillow。')
            return None
        self._build()
        self.top.grab_set()
        self.root.wait_window(self.top)   # 模态，直到确认/取消
        return self.result

    # ---------------- 构建 ----------------
    def _build(self):
        scr_w = self.root.winfo_screenwidth()
        scr_h = self.root.winfo_screenheight()
        scale = min(scr_w * 0.84 / max(1, self.iw),
                    scr_h * 0.78 / max(1, self.ih), self.max_scale)
        self.scale = max(0.05, scale)
        disp_w = max(1, int(round(self.iw * self.scale)))
        disp_h = max(1, int(round(self.ih * self.scale)))
        win_w, win_h = disp_w + 16, disp_h + 76
        pos_x, pos_y = (scr_w - win_w) // 2, max(0, (scr_h - win_h) // 2)

        self.top = tk.Toplevel(self.root)
        self.top.title(self.title)
        self.top.geometry(f'{win_w}x{win_h}+{pos_x}+{pos_y}')
        self.top.attributes('-topmost', True)
        self.top.configure(bg='#111111')
        self.top.protocol('WM_DELETE_WINDOW', self._cancel)

        rgb = cv2.cvtColor(cv2.resize(self.image, (disp_w, disp_h)),
                           cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas = tk.Canvas(self.top, width=disp_w, height=disp_h,
                                highlightthickness=0, cursor='crosshair')
        self.canvas.place(x=8, y=30)
        self.canvas.create_image(0, 0, image=self._photo, anchor='nw')

        tk.Label(self.top, text=self.hint, fg='#dddddd', bg='#111111',
                 font=('Microsoft YaHei', 10)).place(x=8, y=5)
        self.info_var = tk.StringVar(value='拖拽鼠标框选')
        tk.Label(self.top, textvariable=self.info_var, fg='#8bc34a',
                 bg='#111111', font=('Consolas', 10)).place(
            relx=1.0, x=-10, y=5, anchor='ne')
        tk.Button(self.top, text='确认 (Enter)', width=12,
                  command=self._confirm).place(
            relx=1.0, x=-104, rely=1.0, y=-8, anchor='se')
        tk.Button(self.top, text='取消 (Esc)', width=10,
                  command=self._cancel).place(
            relx=1.0, x=-8, rely=1.0, y=-8, anchor='se')

        # 选区外压暗（Tk 用 stipple 模拟半透明）
        self.dim = [self.canvas.create_rectangle(0, 0, 0, 0, fill='#000000',
                                                 stipple='gray50', outline='')
                    for _ in range(4)]
        self.rect = self.canvas.create_rectangle(0, 0, 0, 0,
                                                 outline='#00e676', width=2)
        self.handles = [self.canvas.create_rectangle(0, 0, 0, 0, fill='#00e676',
                                                     outline='#004d40')
                        for _ in range(8)]

        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_motion)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Double-Button-1>', lambda e: self._confirm())
        self.canvas.bind('<Button-3>', lambda e: self._cancel())
        self.top.bind('<Return>', lambda e: self._confirm())
        self.top.bind('<Escape>', lambda e: self._cancel())
        self.top.focus_force()

    # ---------------- 选区读写 ----------------
    def _set_selection(self, x0, y0, x1, y1, redraw=True):
        """以快照像素坐标设置选区（自动排序 + 裁剪到图像内）"""
        x0, x1 = sorted((int(x0), int(x1)))
        y0, y1 = sorted((int(y0), int(y1)))
        x0, x1 = max(0, min(self.iw, x0)), max(0, min(self.iw, x1))
        y0, y1 = max(0, min(self.ih, y0)), max(0, min(self.ih, y1))
        self.sel = (x0, y0, x1, y1)
        if redraw:
            self._redraw()

    def _clear_selection(self):
        self.sel = None
        self._redraw()

    def _to_img(self, mx, my):
        """显示坐标 -> 快照像素坐标"""
        px = int(round(mx / self.scale))
        py = int(round(my / self.scale))
        return max(0, min(self.iw, px)), max(0, min(self.ih, py))

    def _disp_rect(self):
        x0, y0, x1, y1 = self.sel
        return (x0 * self.scale, y0 * self.scale,
                x1 * self.scale, y1 * self.scale)

    # ---------------- 绘制 ----------------
    def _redraw(self):
        dw, dh = self.iw * self.scale, self.ih * self.scale
        if self.sel is None:
            for item in self.dim:
                self.canvas.coords(item, 0, 0, 0, 0)
            self.canvas.coords(self.rect, 0, 0, 0, 0)
            for h in self.handles:
                self.canvas.coords(h, 0, 0, 0, 0)
            self.info_var.set('拖拽鼠标框选')
            return

        x0, y0, x1, y1 = self._disp_rect()
        w, h = x1 - x0, y1 - y0
        self.canvas.coords(self.rect, x0, y0, x1, y1)
        # 上下左右四块压暗区域
        for item, box in zip(self.dim, [
                (0, 0, dw, y0),                 # 上
                (0, y1, dw, dh),                # 下
                (0, y0, x0, y1),                # 左
                (x1, y0, dw, y1)]):             # 右
            self.canvas.coords(item, *box)

        pts = self._handle_points(x0, y0, x1, y1)
        for item, name in zip(self.handles,
                              ('nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w')):
            hx, hy = pts[name]
            r = self.HANDLE // 2
            self.canvas.coords(item, hx - r, hy - r, hx + r, hy + r)

        sx0, sy0, sx1, sy1 = self.sel
        self.info_var.set(f'({sx0},{sy0})  {sx1 - sx0}×{sy1 - sy0}')

    @staticmethod
    def _handle_points(x0, y0, x1, y1):
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        return {'nw': (x0, y0), 'n': (mx, y0), 'ne': (x1, y0), 'e': (x1, my),
                'se': (x1, y1), 's': (mx, y1), 'sw': (x0, y1), 'w': (x0, my)}

    def _handle_at(self, mx, my):
        """命中哪个手柄；返回 None 表示没命中"""
        if self.sel is None:
            return None
        for name, (hx, hy) in self._handle_points(*self._disp_rect()).items():
            if abs(mx - hx) <= self.HANDLE and abs(my - hy) <= self.HANDLE:
                return name
        return None

    # ---------------- 交互 ----------------
    def _on_press(self, e):
        handle = self._handle_at(e.x, e.y)
        if handle:
            self._drag = ('resize', handle, e.x, e.y, self.sel)
            return
        if self.sel is not None:
            x0, y0, x1, y1 = self._disp_rect()
            if x0 <= e.x <= x1 and y0 <= e.y <= y1:
                self._drag = ('move', e.x, e.y, self.sel)
                return
        px, py = self._to_img(e.x, e.y)
        self._set_selection(px, py, px, py, redraw=False)
        self._drag = ('new', e.x, e.y, None)

    def _on_motion(self, e):
        if not self._drag:
            return
        kind = self._drag[0]
        if kind == 'new':
            _, sx, sy, _ = self._drag
            px, py = self._to_img(sx, sy)
            qx, qy = self._to_img(e.x, e.y)
            self._set_selection(px, py, qx, qy)
        elif kind == 'move':
            _, sx, sy, orig = self._drag
            dx = int(round((e.x - sx) / self.scale))
            dy = int(round((e.y - sy) / self.scale))
            x0, y0, x1, y1 = orig
            w, h = x1 - x0, y1 - y0
            nx0 = max(0, min(self.iw - w, x0 + dx))
            ny0 = max(0, min(self.ih - h, y0 + dy))
            self._set_selection(nx0, ny0, nx0 + w, ny0 + h)
        else:
            _, name, sx, sy, orig = self._drag
            x0, y0, x1, y1 = orig
            qx, qy = self._to_img(e.x, e.y)
            if 'n' in name:
                y0 = qy
            if 's' in name:
                y1 = qy
            if 'w' in name:
                x0 = qx
            if 'e' in name:
                x1 = qx
            self._set_selection(x0, y0, x1, y1)

    def _on_release(self, e):
        self._drag = None
        if self.sel is not None:
            x0, y0, x1, y1 = self.sel
            if (x1 - x0) < self.MIN_SIZE or (y1 - y0) < self.MIN_SIZE:
                self._clear_selection()   # 误点一下不算选区

    # ---------------- 确认 / 取消 ----------------
    def _confirm(self):
        if self.sel is None:
            self.info_var.set('请先框选区域')
            return
        x0, y0, x1, y1 = self.sel
        if (x1 - x0) < self.MIN_SIZE or (y1 - y0) < self.MIN_SIZE:
            self.info_var.set(f'选区太小（至少 {self.MIN_SIZE}px）')
            return
        self.result = (x0, y0, x1 - x0, y1 - y0)
        self._close()

    def _cancel(self):
        self.result = None
        self._close()

    def _close(self):
        try:
            self.top.grab_release()
        except Exception:
            pass
        self.top.destroy()


class App:
    def __init__(self, root):
        self.root = root
        root.title(WINDOW_TITLE)
        root.attributes('-topmost', True)
        self.mini = False        # 迷你模式：只保留计数小条
        self._last_title = ''    # 标题里的计数，变化时才刷新
        self._capture_active = False   # 处于冻结帧框选中
        self._was_running = False      # 框选前的运行状态，结束后恢复

        self.cfg = self._load_config()
        self.lock = threading.Lock()
        self.region = list(self.cfg['region']) if self.cfg.get('region') else None
        self.tpl_gray = None
        self.detector = None
        self._running = False
        self._stop = threading.Event()
        self.count = int(self.cfg.get('count', 0))
        self.max_n = int(self.cfg.get('max_n', 0))
        self.cur_n = 0
        self.last_n = None
        self.view = {'score': None, 'frame': None, 'marks': [], 'error': None,
                     'warn': None}
        self._last_gray = None      # 上一帧灰度，用于冻结检测
        self._frozen_since = None
        # 主线程写 / 监控线程读的镜像变量
        self._m_mode = self.cfg.get('mode', 'appear')
        self._m_interval = max(0.05, int(self.cfg.get('interval_ms', 100)) / 1000.0)
        self._m_sound = bool(self.cfg.get('sound', True))
        # 屏幕分辨率/显示器布局变化后，旧坐标会失效，给出提醒
        self._screen_changed = bool(
            self.cfg.get('screen_size')
            and list(self.cfg['screen_size']) != [self.root.winfo_screenwidth(),
                                                  self.root.winfo_screenheight()])

        self._build_ui()
        self._load_template_file(TEMPLATE_PATH, quiet=True)
        if self.region:
            self.region_var.set(
                f'监测区域：({self.region[0]},{self.region[1]}) '
                f'{self.region[2]}×{self.region[3]}')

        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        self._kb = None
        self._bind_hotkeys()
        self._tick()
        root.protocol('WM_DELETE_WINDOW', self._on_close)

    # ---------- 配置 ----------
    def _load_config(self):
        try:
            with open(CONFIG_PATH, encoding='utf-8') as f:
                cfg = json.load(f)
            return {**DEFAULTS, **cfg}
        except Exception:
            return dict(DEFAULTS)

    def _save_config(self):
        data = {
            'region': self.region,
            'threshold': float(self.th_var.get()),
            'mode': self.mode_var.get(),
            'sound': bool(self.sound_var.get()),
            'multiscale': bool(self.ms_var.get()),
            'hotkeys': bool(self.hotkey_var.get()),
            'interval_ms': int(self.itv_var.get()),
            'count': self.count,
            'max_n': self.max_n,
            'topmost': bool(self.topmost_var.get()),
            'tpl_src': self.tpl_src_var.get(),
            'screen_size': [self.root.winfo_screenwidth(),
                            self.root.winfo_screenheight()],
        }
        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _log_event(self, kind, detail=''):
        try:
            new = not os.path.exists(LOG_PATH)
            with open(LOG_PATH, 'a', encoding='utf-8-sig', newline='') as f:
                w = csv.writer(f)
                if new:
                    w.writerow(['时间', '事件', '详情'])
                w.writerow([datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                            kind, detail])
        except Exception:
            pass

    # ---------- 界面 ----------
    def _build_ui(self):
        pad = dict(padx=6, pady=4)
        self.top_frame = top = tk.Frame(self.root)
        top.pack(fill='x', padx=8, pady=(8, 2))
        tk.Label(top, text='计数', font=('Microsoft YaHei', 12)).pack(side='left')
        self.count_var = tk.StringVar(value=str(self.count))
        tk.Label(top, textvariable=self.count_var, font=('Consolas', 34, 'bold'),
                 fg='#d6336c').pack(side='left', padx=10)
        self.extra_var = tk.StringVar(value='')
        tk.Label(top, textvariable=self.extra_var,
                 font=('Microsoft YaHei', 11), fg='#555').pack(side='left')

        self.sel_frame = sel = tk.LabelFrame(self.root, text='1. 设置（先让标识画面显示在屏幕上）')
        sel.pack(fill='x', padx=8, pady=4)
        tk.Button(sel, text='① 框选监测区域', width=15,
                  command=self._select_region).grid(row=0, column=0, **pad)
        self.capture_btn = tk.Button(sel, text='② 截取标识模板', width=15,
                                     command=self._capture_template)
        self.capture_btn.grid(row=0, column=1, **pad)
        tk.Button(sel, text='从图片加载模板', width=14,
                  command=self._load_template_dialog).grid(row=0, column=2, **pad)
        tk.Button(sel, text='测试截图', width=9,
                  command=self._test_grab).grid(row=0, column=3, **pad)
        self.region_var = tk.StringVar(value='监测区域：未设置')
        self.tpl_var = tk.StringVar(value='模板：未设置')
        tk.Label(sel, textvariable=self.region_var, fg='#333').grid(
            row=1, column=0, columnspan=2, sticky='w', **pad)
        tk.Label(sel, textvariable=self.tpl_var, fg='#333').grid(
            row=1, column=2, columnspan=2, sticky='w', **pad)
        # 模板来源：默认区域内截取；想更灵活可切到整屏任意位置
        tk.Label(sel, text='模板来源').grid(row=2, column=0, sticky='w', **pad)
        self.tpl_src_var = tk.StringVar(
            value=self.cfg.get('tpl_src', TPL_SRC_REGION))
        tk.OptionMenu(sel, self.tpl_src_var, TPL_SRC_REGION, TPL_SRC_SCREEN,
                      command=lambda _v: self._tpl_src_changed()).grid(
            row=2, column=1, columnspan=2, sticky='w', **pad)
        self._tpl_src_changed(save=False)

        self.run_frame = run = tk.LabelFrame(self.root, text='2. 运行')
        run.pack(fill='x', padx=8, pady=4)
        self.run_btn = tk.Button(run, text='开始监测', width=12,
                                 command=self._toggle_run)
        self.run_btn.grid(row=0, column=0, **pad)
        tk.Button(run, text='重置计数', width=10,
                  command=self._reset_count).grid(row=0, column=1, **pad)
        tk.Button(run, text='迷你模式', width=10,
                  command=self._toggle_mini).grid(row=0, column=2, **pad)
        tk.Button(run, text='使用说明', width=10,
                  command=self._show_help).grid(row=0, column=3, **pad)
        tk.Button(run, text='退出', width=6,
                  command=self._on_close).grid(row=0, column=4, **pad)

        self.opt_frame = opt = tk.LabelFrame(self.root, text='参数')
        opt.pack(fill='x', padx=8, pady=4)
        self.mode_var = tk.StringVar(value=self.cfg.get('mode', 'appear'))
        tk.Radiobutton(opt, text='出现即计数', variable=self.mode_var,
                       value='appear', command=self._mode_changed).grid(
            row=0, column=0, sticky='w', **pad)
        tk.Radiobutton(opt, text='统计可见数量', variable=self.mode_var,
                       value='multi', command=self._mode_changed).grid(
            row=0, column=1, sticky='w', **pad)
        tk.Label(opt, text='相似度阈值').grid(row=1, column=0, sticky='w', **pad)
        self.th_var = tk.DoubleVar(value=float(self.cfg.get('threshold', 0.8)))
        tk.Scale(opt, from_=0.5, to=0.98, resolution=0.01, orient='horizontal',
                 length=190, variable=self.th_var,
                 command=self._threshold_changed).grid(
            row=1, column=1, columnspan=2, sticky='w', **pad)
        self.sound_var = tk.BooleanVar(value=bool(self.cfg.get('sound', True)))
        tk.Checkbutton(opt, text='计数提示音',
                       variable=self.sound_var).grid(row=2, column=0, sticky='w', **pad)
        self.ms_var = tk.BooleanVar(value=bool(self.cfg.get('multiscale', False)))
        tk.Checkbutton(opt, text='多尺度匹配(慢)', variable=self.ms_var,
                       command=self._rebuild_detector).grid(
            row=2, column=1, sticky='w', **pad)
        self.hotkey_var = tk.BooleanVar(value=bool(self.cfg.get('hotkeys', True)))
        self.hk_cb = tk.Checkbutton(opt, text='全局热键 F8/F9/F10',
                                    variable=self.hotkey_var,
                                    command=self._toggle_hotkeys)
        self.hk_cb.grid(row=2, column=2, sticky='w', **pad)
        tk.Label(opt, text='检测间隔(ms)').grid(row=3, column=0, sticky='w', **pad)
        self.itv_var = tk.IntVar(value=int(self.cfg.get('interval_ms', 100)))
        tk.Spinbox(opt, from_=50, to=1000, increment=50, width=7,
                   textvariable=self.itv_var).grid(row=3, column=1, sticky='w', **pad)
        self.topmost_var = tk.BooleanVar(value=bool(self.cfg.get('topmost', True)))
        tk.Checkbutton(opt, text='窗口置顶', variable=self.topmost_var,
                       command=self._apply_topmost).grid(
            row=3, column=2, sticky='w', **pad)

        self.prev_frame = prev = tk.LabelFrame(self.root, text='实时预览')
        prev.pack(fill='both', expand=True, padx=8, pady=4)
        self.canvas = tk.Canvas(prev, width=440, height=248, bg='#1a1a1a',
                                highlightthickness=0)
        self.canvas.pack(padx=4, pady=4)

        self.status_var = tk.StringVar(value='就绪')
        self.status_label = tk.Label(self.root, textvariable=self.status_var,
                                     anchor='w', fg='#666')
        self.status_label.pack(fill='x', padx=10, pady=(0, 6))

        # 迷你条：默认隐藏，切换后只显示计数 + 暂停 + 还原
        self.mini_frame = mf = tk.Frame(self.root)
        tk.Label(mf, textvariable=self.count_var, font=('Consolas', 26, 'bold'),
                 fg='#d6336c').pack(side='left', padx=(8, 6))
        tk.Label(mf, textvariable=self.extra_var, font=('Microsoft YaHei', 10),
                 fg='#555').pack(side='left')
        self.mini_run_btn = tk.Button(mf, text='暂停', width=5,
                                      command=self._toggle_run)
        self.mini_run_btn.pack(side='right', padx=(4, 8))
        tk.Button(mf, text='还原', width=5,
                  command=self._toggle_mini).pack(side='right')

        self._apply_topmost()
        # 右键菜单：X 点不动时的备用出口
        self._menu = tk.Menu(self.root, tearoff=0)
        self._menu.add_command(label='暂停 / 继续', command=self._toggle_run)
        self._menu.add_command(label='重置计数', command=self._reset_count)
        self._menu.add_separator()
        self._menu.add_command(label='退出', command=self._on_close)
        self.root.bind_all('<Button-3>', self._popup_menu)

    def _popup_menu(self, event):
        if getattr(self, '_capture_active', False):   # 框选中不弹菜单
            return
        try:
            self._menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._menu.grab_release()

    # ---------- 迷你模式 / 置顶 ----------
    def _toggle_mini(self):
        self.mini = not self.mini
        if self.mini:
            for f in (self.top_frame, self.sel_frame, self.run_frame,
                      self.opt_frame, self.prev_frame, self.status_label):
                f.pack_forget()
            self.mini_frame.pack(fill='both', expand=True, padx=6, pady=6)
            self.root.geometry('250x70-24+16')   # 右上角
        else:
            self.mini_frame.pack_forget()
            self.top_frame.pack(fill='x', padx=8, pady=(8, 2))
            self.sel_frame.pack(fill='x', padx=8, pady=4)
            self.run_frame.pack(fill='x', padx=8, pady=4)
            self.opt_frame.pack(fill='x', padx=8, pady=4)
            self.prev_frame.pack(fill='both', expand=True, padx=8, pady=4)
            self.status_label.pack(fill='x', padx=10, pady=(0, 6))
            self.root.geometry('')

    def _tpl_src_changed(self, save=True):
        """切换模板来源时同步按钮文案（整屏模式下 ② 一次框选定区域+模板）"""
        screen_mode = self.tpl_src_var.get() == TPL_SRC_SCREEN
        self.capture_btn.config(
            text='② 框选区域+模板' if screen_mode else '② 截取标识模板')
        if save:
            self._save_config()

    def _apply_topmost(self):
        try:
            self.root.attributes('-topmost', bool(self.topmost_var.get()))
        except Exception:
            pass

    # ---------- 区域 / 模板 ----------
    def _select_region(self):
        """在「屏幕冻结帧」上框选监测区域"""
        if self._capture_active:
            return
        try:
            snap, origin = self._snapshot_screen()
        except Exception as e:
            self.status_var.set(f'截屏失败: {e}')
            return
        self._enter_capture_mode()
        try:
            hint = '拖拽框选；可拖动选区或边角手柄微调；Enter 确认 / Esc 取消'
            rect = SnapshotSelector(self.root, snap, '框选监测区域', hint).run()
            if not rect:
                self.status_var.set('已取消框选')
                return
            x, y, w, h = rect
            with self.lock:
                self.region = [origin[0] + x, origin[1] + y, w, h]
                self.view = {'score': None, 'frame': None, 'marks': [],
                             'error': None, 'warn': None}
                self._last_gray = None
                self._frozen_since = None
            self.region_var.set(f'监测区域：({origin[0] + x},{origin[1] + y}) {w}×{h}')
            self._save_config()
        finally:
            self._exit_capture_mode()

    def _capture_template(self):
        """截取标识模板。

        来源二选一：
          - TPL_SRC_REGION：在「监测区域的冻结帧」上裁切 → 同源同尺度，最稳
          - TPL_SRC_SCREEN：在「整屏冻结帧」上框选，框选范围**同时作为监测区域**
            （位置与大小完全一致），模板即该范围 → 一次框选定稿，无需再点①
        """
        if self._capture_active:
            return
        with self.lock:
            region = list(self.region) if self.region else None
        src = self.tpl_src_var.get()
        screen_mode = (src == TPL_SRC_SCREEN)
        if not screen_mode and region is None:
            messagebox.showinfo('提示', '请先点「① 框选监测区域」，再截取模板。\n'
                                        '（或把「模板来源」切到「整屏任意位置」，'
                                        '框选范围会自动成为监测区域）')
            return
        try:
            if screen_mode:
                snap, origin = self._snapshot_screen()
                hint = ('框住要监测的范围（已冻结）——它会同时成为监测区域与模板\n'
                        '越贴合标识越好；确认后位置与大小将覆盖原监测区域')
                title = '框选监测区域 + 标识模板（整屏任意位置）'
            else:
                snap, origin = self._snapshot_region(region)
                hint = (f'来源：监测区域 {region[2]}×{region[3]}（已冻结）　'
                        '框住标识本身，越紧凑越准')
                title = '截取标识模板'
        except Exception as e:
            what = '整屏' if screen_mode else '监测区域'
            self.status_var.set(f'截取{what}失败（区域可能已越界）: {e}')
            return
        self._enter_capture_mode()
        try:
            rect = SnapshotSelector(self.root, snap, title, hint).run()
            if not rect:
                self.status_var.set('已取消截取')
                return
            x, y, w, h = rect
            crop = snap[y:y + h, x:x + w]
            if crop.size == 0:
                self.status_var.set('选区无效')
                return
            note = ''
            if screen_mode:
                # 框选范围即监测区域（位置 + 大小完全一致）
                new_region = [origin[0] + x, origin[1] + y, w, h]
                with self.lock:
                    self.region = new_region
                    self.view = {'score': None, 'frame': None, 'marks': [],
                                 'error': None, 'warn': None}
                    self._last_gray = None
                    self._frozen_since = None
                self.region_var.set(
                    f'监测区域：({new_region[0]},{new_region[1]}) {w}×{h}')
                mon = self._monitor_at(new_region[0] + w // 2, new_region[1] + h // 2)
                note = ('监测区域已同步为框选范围%s；模板＝整块区域，'
                        '相似度是整块比对，建议把「相似度阈值」降到 0.5~0.7，'
                        '想更稳可切回「监测区域内」再框一次更小的标识'
                        % (('（显示器 %d）' % mon) if mon else ''))
            else:
                # 区域内模式：模板必须能落在监测区域内，且不能大到失去区分度
                ok, _warn, note = self._validate_template(w, h, region)
                if not ok:
                    self.status_var.set('模板不可用：' + note)
                    messagebox.showwarning('模板不可用', note)
                    return          # 不覆盖已有模板
            try:
                cv2.imencode('.png', crop)[1].tofile(TEMPLATE_PATH)
            except Exception:
                pass
            self._apply_template(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
                                 f'{w}×{h} (template.png)')
            self._save_config()
            self.status_var.set('模板已截取 — ' + note if note else '模板已截取')
        finally:
            self._exit_capture_mode()

    @staticmethod
    def _validate_template(w, h, region):
        """校验模板尺寸与监测区域的关系。

        返回 (是否可用, 是否告警, 提示语)：
          - 模板比监测区域还大 → 模板匹配在数学上不可能成功，直接拒绝
          - 模板占区域面积 > 60% → 相似度会接近 1，计数失去区分度，告警但允许
        """
        if region is None:
            return True, False, ''
        rw, rh = max(1, int(region[2])), max(1, int(region[3]))
        if w > rw or h > rh:
            return False, True, (
                '模板 %d×%d 比监测区域 %d×%d 还大，模板匹配无法执行。\n'
                '请改框更小的标识，或先把监测区域放大。' % (w, h, rw, rh))
        ratio = (w * h) / float(rw * rh)
        if ratio > 0.6:
            return True, True, '模板偏大（占监测区域约 %d%%），建议只框标识本身' % int(ratio * 100)
        return True, False, ''

    @staticmethod
    def _monitor_at(x, y):
        """返回包含该点的显示器序号（1 起），取不到返回 None"""
        try:
            with mss.MSS() as sct:
                for i, mon in enumerate(sct.monitors[1:], start=1):
                    if (mon['left'] <= x < mon['left'] + mon['width']
                            and mon['top'] <= y < mon['top'] + mon['height']):
                        return i
        except Exception:
            pass
        return None

    # ---------- 冻结帧快照 ----------
    def _snapshot_screen(self):
        """抓取鼠标所在显示器的冻结帧，返回 (图像, (left, top))"""
        with mss.MSS() as sct:
            mon = self._monitor_under_cursor(sct)
            raw = sct.grab(mon)
            return (cv2.cvtColor(np.asarray(raw), cv2.COLOR_BGRA2BGR),
                    (int(mon['left']), int(mon['top'])))

    def _snapshot_region(self, region):
        """抓取监测区域的冻结帧，返回 (图像, (left, top))"""
        img = _grab_region(tuple(region))
        return img, (int(region[0]), int(region[1]))

    @staticmethod
    def _monitor_under_cursor(sct):
        """按鼠标位置选择显示器；取不到就用主显示器（mss.monitors[1]）"""
        try:
            pt = _POINT()
            if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
                for mon in sct.monitors[1:]:
                    if (mon['left'] <= pt.x < mon['left'] + mon['width']
                            and mon['top'] <= pt.y < mon['top'] + mon['height']):
                        return mon
        except Exception:
            pass
        return sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]

    # ---------- 截取模式（期间暂停监测，避免抓到遮罩本身） ----------
    def _enter_capture_mode(self):
        self._capture_active = True
        with self.lock:
            self._was_running = self._running
            self._running = False
        self.run_btn.config(text='开始监测')
        self.mini_run_btn.config(text='开始')

    def _exit_capture_mode(self):
        with self.lock:
            self._running = self._was_running
            self._last_gray = None
            self._frozen_since = None
        text = '暂停' if self._was_running else '开始监测'
        self.run_btn.config(text=text)
        self.mini_run_btn.config(text='暂停' if self._was_running else '开始')
        self._capture_active = False

    def _load_template_dialog(self):
        p = filedialog.askopenfilename(
            title='选择标识模板图片',
            filetypes=[('图片', '*.png *.jpg *.jpeg *.bmp'), ('所有文件', '*.*')])
        if not p:
            return
        img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror('错误', '无法读取该图片')
            return
        try:
            ok, buf = cv2.imencode('.png', img)
            if ok:
                buf.tofile(TEMPLATE_PATH)
        except Exception:
            pass
        self._apply_template(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                             f'{img.shape[1]}×{img.shape[0]} (已存为 template.png)')

    def _load_template_file(self, path, quiet=False):
        if os.path.exists(path):
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                self._apply_template(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                     f'{img.shape[1]}×{img.shape[0]} (template.png)')
                return True
        if not quiet:
            self.status_var.set('未找到模板文件')
        return False

    def _apply_template(self, gray, desc):
        with self.lock:
            self.tpl_gray = gray
            self.detector = Detector(gray,
                                     threshold=float(self.th_var.get()),
                                     multiscale=bool(self.ms_var.get()))
        self.tpl_var.set(f'模板：{desc}')
        self._save_config()

    def _rebuild_detector(self):
        if self.tpl_gray is None:
            return
        with self.lock:
            self.detector = Detector(self.tpl_gray,
                                     threshold=float(self.th_var.get()),
                                     multiscale=bool(self.ms_var.get()))

    def _threshold_changed(self, _=None):
        with self.lock:
            if self.detector is not None:
                self.detector.set_threshold(float(self.th_var.get()))

    def _test_grab(self):
        with self.lock:
            region = self.region
        if region is None:
            self.status_var.set('请先框选监测区域')
            return
        try:
            img = _grab_region(tuple(region))
        except Exception as e:
            self.status_var.set(f'截屏失败: {e}')
            return
        with self.lock:
            self.view = {'score': None, 'frame': img, 'marks': [], 'error': None}

    # ---------- 运行控制 ----------
    def _toggle_run(self):
        with self.lock:
            self._running = not self._running
            running = self._running
        self.run_btn.config(text='暂停' if running else '开始监测')
        self.mini_run_btn.config(text='暂停' if running else '开始')

    def _reset_count(self):
        with self.lock:
            self.count = 0
            self.max_n = 0
            self.cur_n = 0
            self.last_n = None
            if self.detector is not None:
                self.detector.visible = False
                self.detector.hit = 0
                self.detector.miss = 0
        self._log_event('重置', '')
        self._save_config()

    def _mode_changed(self):
        with self.lock:
            self._m_mode = self.mode_var.get()
            self.last_n = None
            self.cur_n = 0

    # ---------- 热键 ----------
    def _bind_hotkeys(self):
        if not self.hotkey_var.get():
            return
        try:
            import keyboard
            self._kb = keyboard
            keyboard.add_hotkey('f8', lambda: self.root.after(0, self._toggle_run))
            keyboard.add_hotkey('f9', lambda: self.root.after(0, self._reset_count))
            keyboard.add_hotkey('f10', lambda: self.root.after(0, self._on_close))
        except Exception:
            self._kb = None
            self.hotkey_var.set(False)
            try:
                self.hk_cb.configure(state='disabled', text='热键不可用')
            except Exception:
                pass

    def _toggle_hotkeys(self):
        if self._kb is not None:
            try:
                self._kb.unhook_all()
            except Exception:
                pass
            self._kb = None
        if self.hotkey_var.get():
            self._bind_hotkeys()

    # ---------- 监控线程 ----------
    def _monitor(self):
        sct = mss.MSS()
        while not self._stop.is_set():
            with self.lock:
                running = self._running
                region = self.region
                detector = self.detector
                mode = self._m_mode
                interval = self._m_interval
                sound = self._m_sound
            if not running or region is None or detector is None:
                time.sleep(0.12)
                continue
            frame = None
            err = None
            warn = None
            score = None
            marks = []
            try:
                frame = _grab_region_with(sct, region)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                warn = self._check_frozen(gray)
                if mode == 'multi':
                    peaks = detector.match_multi(gray)
                    n = len(peaks)
                    if peaks:
                        score = peaks[0][2]
                    marks = [(x, y, detector.tw, detector.th) for x, y, _ in peaks]
                    with self.lock:
                        self.cur_n = n
                        if n > self.max_n:
                            self.max_n = n
                        if n != self.last_n:
                            self.last_n = n
                            self._log_event('可见数量', f'n={n}')
                else:
                    score, loc, s = detector.match_best(gray)
                    ev = detector.update(score)
                    if loc is not None:
                        marks = [(loc[0], loc[1],
                                  int(detector.tw * s), int(detector.th * s))]
                    if ev == 'count':
                        with self.lock:
                            self.count += 1
                        self._log_event('计数', f'score={score:.3f}')
                        if sound and winsound is not None:
                            try:
                                winsound.Beep(1200, 120)
                            except Exception:
                                pass
            except Exception as e:
                err = f'错误: {e}'
            with self.lock:
                v = {'score': score, 'marks': marks, 'error': err, 'warn': warn,
                     'frame': frame if frame is not None else self.view.get('frame')}
                self.view = v
            time.sleep(interval)

    def _check_frozen(self, gray):
        """画面长时间完全不变或纯色 → 大概率被遮挡/游戏暂停，给出提示"""
        now = time.time()
        msg = None
        if float(gray.std()) < 2.0:
            msg = '画面纯色/黑屏，可能已被遮挡'
        else:
            prev = self._last_gray
            if prev is not None and prev.shape == gray.shape:
                if float(cv2.absdiff(gray, prev).mean()) < 0.5:
                    if self._frozen_since is None:
                        self._frozen_since = now
                    elif now - self._frozen_since > 5.0:
                        msg = '画面 5 秒无变化，可能被遮挡或游戏已暂停'
                else:
                    self._frozen_since = None
            else:
                self._frozen_since = None
        self._last_gray = gray
        return msg

    # ---------- 界面刷新 ----------
    def _tick(self):
        try:
            itv = int(self.itv_var.get())
        except Exception:
            itv = 100
        with self.lock:
            self._m_interval = max(0.05, itv / 1000.0)
            self._m_mode = self.mode_var.get()
            self._m_sound = bool(self.sound_var.get())
            view = dict(self.view)
            count = self.count
            max_n = self.max_n
            cur_n = self.cur_n

        if self._m_mode == 'multi':
            self.count_var.set(str(cur_n))
            self.extra_var.set(f'峰值 {max_n}')
            shown = cur_n
        else:
            self.count_var.set(str(count))
            self.extra_var.set('')
            shown = count

        # 计数写进窗口标题：最小化到任务栏也能看到数字
        title = '%s — 计数 %s' % (WINDOW_TITLE, shown)
        if title != self._last_title:
            self._last_title = title
            self.root.title(title)

        parts = ['监测中' if self._running else '已暂停']
        if view.get('error'):
            parts.append(view['error'])
        elif view.get('score') is not None:
            parts.append(f"相似度 {view['score']:.3f}")
        if view.get('warn'):
            parts.append('⚠ ' + view['warn'])
        if self.tpl_gray is None:
            parts.append('缺模板')
        if self.region is None:
            parts.append('缺监测区域')
        if self._screen_changed:
            parts.append('⚠ 屏幕分辨率已变化，建议重新框选')
        self.status_var.set(' | '.join(parts))

        if PIL_OK:
            self._draw_preview(view)
        self.root.after(120, self._tick)

    def _draw_preview(self, view):
        self.canvas.delete('all')
        frame = view.get('frame')
        if frame is None:
            self.canvas.create_text(
                220, 124,
                text='暂无画面（设置区域后点「开始监测」或「测试截图」）',
                fill='#999999', font=('Microsoft YaHei', 10))
            return
        h, w = frame.shape[:2]
        scale = min(440 / w, 248 / h, 2.0)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        img = cv2.cvtColor(cv2.resize(frame, (nw, nh)), cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(img))
        self._photo = photo
        ox, oy = (440 - nw) // 2, (248 - nh) // 2
        self.canvas.create_image(ox, oy, image=photo, anchor='nw')
        for (x, y, mw, mh) in view.get('marks', []):
            self.canvas.create_rectangle(
                ox + x * scale, oy + y * scale,
                ox + (x + mw) * scale, oy + (y + mh) * scale,
                outline='#00e676', width=2)

    def _show_help(self):
        messagebox.showinfo('使用说明', (
            '使用步骤：\n'
            '1. 让屏幕上出现你要监测的标识（如游戏内触发画面）。\n'
            '2. 点「① 框选监测区域」，拖拽框住标识可能出现的范围。\n'
            '3. 点「② 截取标识模板」，拖拽框住标识本身（越紧凑越准），\n'
            '    也可点「从图片加载模板」选一张图片。\n'
            '4. 点「开始监测」。标识出现 → 计数 +1，并写入 counter_log.csv。\n\n'
            '提示：\n'
            '· 状态栏实时显示相似度：出现标识时应明显高于未出现时；\n'
            '  区分不开就调「相似度阈值」（默认 0.80）。\n'
            '· 模板应与屏幕上实际显示大小一致（同分辨率/缩放）；\n'
            '  大小会变化就勾选「多尺度匹配」。\n'
            '· 同一标识持续显示只计一次；1 秒内重复触发会合并。\n'
            '  标识一闪而过时，把「检测间隔」调小（如 50ms）。\n'
            '· 「统计可见数量」模式：实时统计区域内标识出现了几个并记峰值。\n'
            '· 框选时会先冻结一帧画面（同系统截图工具）：拖拽框选、拖动选区、\n'
            '  拖八个手柄微调，Enter 或「确认」生效，Esc / 右键 / 「取消」放弃。\n'
            '· 「模板来源」可选：\n'
            '  · 监测区域内（默认，同源同尺度最稳）：先①框区域，再②在区域内框标识。\n'
            '  · 整屏任意位置：②一次框选定稿 —— 框选范围会同时成为监测区域\n'
            '    （位置与大小完全一致）并作为模板；适合「我框哪就监测哪」。\n'
            '    代价：模板＝整块区域，相似度是整块比对，建议阈值降到 0.5~0.7；\n'
            '    想更稳就切回「监测区域内」再框一次更小的标识。\n'
            '· 区域内模式下模板不能大于监测区域（会被拒绝），占区域 >60% 会告警。\n'
            '· 框选期间会自动暂停监测，结束后自动恢复。\n'
            '· 挡视野怎么办：点「迷你模式」缩成右上角小条；或点窗口最小化\n'
            '  （最小化后计数照常进行，数字会显示在任务栏标题里）；\n'
            '  不想置顶就取消勾选「窗口置顶」。\n'
            '· 前提是监测区域在屏幕上可见：被别的窗口完全遮住、游戏最小化、\n'
            '  锁屏息屏都会导致抓不到画面（状态栏会给出 ⚠ 提示）。\n'
            '· F8 暂停/继续，F9 重置，F10 退出（窗口被游戏挡住时用得上）。\n'
            '  实在关不掉：任务管理器结束 pythonw.exe，或命令行执行\n'
            '  taskkill /F /IM pythonw.exe\n'
            '· 配置/日志/模板均在本程序同目录。\n'
            '· 限制：仅支持主显示器；模板需在当前分辨率下截取。'))

    def _on_close(self):
        """关闭：多处入口（X / 退出按钮 / F10 / 右键菜单）都走这里"""
        if getattr(self, '_closing', False):
            return                      # 重复触发直接忽略
        self._closing = True
        try:
            self._save_config()
            self._stop.set()
        except Exception:
            pass
        if self._kb is not None:
            try:
                self._kb.unhook_all()
            except Exception:
                pass
            self._kb = None
        # 兜底：2 秒后仍未退出（线程卡住 / destroy 失败）就强制结束
        try:
            watchdog = threading.Timer(2.0, lambda: os._exit(0))
            watchdog.daemon = True
            watchdog.start()
        except Exception:
            pass
        try:
            self.root.grab_release()    # 万一残留模态抓取
        except Exception:
            pass
        try:
            self.root.destroy()         # 先销毁窗口
        except Exception:
            os._exit(0)
        try:
            self.root.quit()            # 再退出 mainloop（防止循环没退出）
        except Exception:
            os._exit(0)


def selftest():
    rng = np.random.default_rng(7)
    # 1) 单实例匹配 + 状态机
    scene = rng.integers(0, 256, (300, 420), dtype=np.uint8)
    tpl = scene[60:60 + 44, 90:90 + 64].copy()
    det = Detector(tpl, threshold=0.8, multiscale=False,
                   confirm=2, miss_frames=2, cooldown=0.0)
    score, loc, s = det.match_best(scene)
    assert score > 0.95, f'匹配分数过低: {score}'
    assert loc == (90, 60), f'定位错误: {loc}'
    evs = [det.update(score) for _ in range(4)]
    assert evs.count('count') == 1, f'出现应计数1次: {evs}'
    evs2 = [det.update(score) for _ in range(4)]
    assert not any(evs2), f'持续存在不应重复计数: {evs2}'
    noise = rng.integers(0, 256, scene.shape, dtype=np.uint8)
    sn, _, _ = det.match_best(noise)
    for _ in range(6):
        det.update(sn)
    assert det.visible is False, '未正确判定消失'
    evs4 = [det.update(score) for _ in range(4)]
    assert evs4.count('count') == 1, f'再次出现应再计数1次: {evs4}'
    # 2) 多实例
    scene2 = rng.integers(0, 256, (300, 420), dtype=np.uint8)
    scene2[40:40 + 44, 30:30 + 64] = tpl
    scene2[200:200 + 44, 250:250 + 64] = tpl
    peaks = det.match_multi(scene2)
    assert len(peaks) == 2, f'多实例应找到2个: {peaks}'
    # 3) 截屏链路
    img = _grab_region((0, 0, 120, 90))
    assert img.shape[0] == 90 and img.shape[1] == 120, f'截屏尺寸异常: {img.shape}'
    # 4) 画面冻结检测
    class _FrozenStub:
        _last_gray = None
        _frozen_since = None
        _check_frozen = App._check_frozen

    stub = _FrozenStub()
    flat = np.full((40, 60), 5, dtype=np.uint8)          # 纯色/黑屏
    assert stub._check_frozen(flat) is not None, '纯色画面未告警'
    a = rng.integers(0, 256, (40, 60), dtype=np.uint8)
    assert stub._check_frozen(a) is None, '正常画面误告警'
    stub._frozen_since = time.time() - 10                 # 模拟已冻结 10 秒
    assert stub._check_frozen(a.copy()) is not None, '长时间无变化未告警'

    # 5) 冻结帧选区器：确认路径 + 取消路径 + 越界裁剪
    img = rng.integers(0, 256, (120, 200, 3), dtype=np.uint8)
    root = tk.Tk()
    root.withdraw()

    def drive(selector, action):
        def _do():
            getattr(selector, action[0])(*action[1:])
        root.after(120, _do)

    sel = SnapshotSelector(root, img, 't', 'hint')
    drive(sel, ('_set_selection', 10, 20, 60, 50))
    drive(sel, ('_confirm',))
    assert sel.run() == (10, 20, 50, 30), sel.run()

    sel2 = SnapshotSelector(root, img, 't', 'hint')
    drive(sel2, ('_set_selection', 10, 20, 60, 50))
    drive(sel2, ('_cancel',))
    assert sel2.run() is None, '取消后应返回 None'

    sel3 = SnapshotSelector(root, img, 't', 'hint')
    drive(sel3, ('_set_selection', -30, -30, 9999, 9999))   # 越界应被裁剪
    drive(sel3, ('_confirm',))
    assert sel3.run() == (0, 0, 200, 120), sel3.run()
    root.destroy()

    # 6) 真实构建完整界面（含 App 初始化流程）
    guitest()
    print('SELFTEST PASS')


def _use_temp_paths():
    """把配置/模板/日志切到临时目录，返回还原函数（测试里避免污染真实文件）"""
    global CONFIG_PATH, TEMPLATE_PATH, LOG_PATH
    import tempfile
    saved = (CONFIG_PATH, TEMPLATE_PATH, LOG_PATH)
    tmp = tempfile.mkdtemp(prefix='scounter_test_')
    CONFIG_PATH = os.path.join(tmp, 'config.json')
    TEMPLATE_PATH = os.path.join(tmp, 'template.png')
    LOG_PATH = os.path.join(tmp, 'counter_log.csv')

    def restore():
        global CONFIG_PATH, TEMPLATE_PATH, LOG_PATH
        CONFIG_PATH, TEMPLATE_PATH, LOG_PATH = saved

    return restore


def guitest():
    """构建整个界面并立即销毁；临时接管配置路径，不污染真实 config.json"""
    restore = _use_temp_paths()
    try:
        root = tk.Tk()
        root.withdraw()
        app = App(root)
        root.update()
        root.update_idletasks()
        # 迷你模式来回切换，验证布局不报错
        app._toggle_mini()
        root.update()
        app._toggle_mini()
        root.update()
        app._stop.set()
        root.destroy()
    finally:
        restore()


def capturetest():
    """模板截取全链路：监测区域快照 -> 裁切 -> 生效 -> 落盘（跳过鼠标交互）"""
    restore = _use_temp_paths()
    orig_run = SnapshotSelector.run
    SnapshotSelector.run = lambda self: (10, 5, 40, 25)
    try:
        root = tk.Tk()
        root.withdraw()
        app = App(root)
        with app.lock:
            app.region = [100, 100, 200, 120]
        app._capture_template()
        assert app.tpl_gray is not None, '模板未生成'
        assert app.tpl_gray.shape == (25, 40), app.tpl_gray.shape
        assert os.path.exists(TEMPLATE_PATH), '模板未落盘'
        # 取消路径
        SnapshotSelector.run = lambda self: None
        keep = app.tpl_gray
        app._capture_template()
        assert app.tpl_gray is keep, '取消后不应覆盖模板'
        app._stop.set()
        root.destroy()
        print('CAPTURE PASS')
        return True
    finally:
        SnapshotSelector.run = orig_run
        restore()


def tpltest():
    """模板来源：区域内截取 / 整屏框选（区域=框选范围）/ 尺寸校验 / 取消不覆盖"""
    restore = _use_temp_paths()
    orig_run = SnapshotSelector.run
    orig_warn, orig_info = messagebox.showwarning, messagebox.showinfo
    messagebox.showwarning = lambda *a, **k: None          # 测试时不弹窗阻塞
    messagebox.showinfo = lambda *a, **k: None
    try:
        # 1) 纯函数：尺寸校验
        assert App._validate_template(150, 20, [0, 0, 100, 100])[0] is False, '大于区域应被拒绝'
        ok, warn, _ = App._validate_template(80, 80, [0, 0, 100, 100])
        assert ok and warn, '占 64% 应告警但允许'
        ok, warn, _ = App._validate_template(20, 20, [0, 0, 100, 100])
        assert ok and not warn, '小模板不应告警'
        assert App._validate_template(50, 50, None)[0] is True, '无区域时不做尺寸限制'

        root = tk.Tk()
        root.withdraw()
        app = App(root)

        # 2) 整屏模式：框选范围 = 监测区域（位置+大小完全一致），模板即该范围
        _snap, origin = app._snapshot_screen()
        with app.lock:
            app.region = [100, 100, 200, 150]        # 旧区域，应被覆盖
        app.tpl_src_var.set(TPL_SRC_SCREEN)
        SnapshotSelector.run = lambda self: (10, 10, 30, 20)
        app._capture_template()
        assert app.tpl_gray is not None, '整屏模式应能截取模板'
        assert app.tpl_gray.shape == (20, 30), app.tpl_gray.shape
        assert app.region == [origin[0] + 10, origin[1] + 10, 30, 20], \
            '监测区域未同步为框选范围: %r' % (app.region,)
        assert '30×20' in app.region_var.get(), app.region_var.get()

        # 3) 整屏模式：原本没有监测区域也能一次框选定稿
        with app.lock:
            app.region = None
        SnapshotSelector.run = lambda self: (5, 5, 16, 12)
        app._capture_template()
        assert app.region == [origin[0] + 5, origin[1] + 5, 16, 12], app.region
        assert app.tpl_gray.shape == (12, 16), app.tpl_gray.shape

        # 4) 整屏模式：取消 → 区域与模板都不动
        keep_region, keep_tpl = list(app.region), app.tpl_gray
        SnapshotSelector.run = lambda self: None
        app._capture_template()
        assert app.region == keep_region, '取消后不应改动监测区域'
        assert app.tpl_gray is keep_tpl, '取消后不应覆盖模板'

        # 5) 区域内模式 + 无区域 → 直接提示并返回，不进入框选
        with app.lock:
            app.region = None
        app.tpl_src_var.set(TPL_SRC_REGION)
        called = []
        SnapshotSelector.run = lambda self: called.append(1) or None
        app._capture_template()
        assert not called, '无监测区域时不应进入框选'

        # 6) 区域内模式：正常路径仍然可用，且不改动监测区域
        with app.lock:
            app.region = [100, 100, 200, 150]
        SnapshotSelector.run = lambda self: (12, 8, 40, 24)
        app._capture_template()
        assert app.tpl_gray.shape == (24, 40), app.tpl_gray.shape
        assert app.region == [100, 100, 200, 150], '区域内模式不应改动监测区域'

        app._stop.set()
        root.destroy()
        print('TPLTEST PASS')
        return True
    finally:
        SnapshotSelector.run = orig_run
        messagebox.showwarning, messagebox.showinfo = orig_warn, orig_info
        restore()


def _ensure_single_instance():
    """只允许一个实例运行，避免多开导致重复计数"""
    if sys.platform != 'win32':
        return True
    try:
        global _MUTEX
        _MUTEX = ctypes.windll.kernel32.CreateMutexW(None, False, 'ScreenCounter_SingleInstance_Mutex')
        return ctypes.windll.kernel32.GetLastError() != 183  # ERROR_ALREADY_EXISTS
    except Exception:
        return True


def _activate_existing_window():
    """把已运行的窗口提到前台，避免用户以为点了没反应"""
    if sys.platform != 'win32':
        return
    # 标题会带实时计数，所以按前缀匹配
    try:
        user32 = ctypes.windll.user32
        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p,
                                             ctypes.c_void_p)
        found = []

        def cb(hwnd, _):
            length = user32.GetWindowTextLengthW(hwnd)
            if length:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                if buf.value.startswith(WINDOW_TITLE):
                    found.append(hwnd)
                    return False
            return True

        user32.EnumWindows(EnumWindowsProc(cb), 0)
        if found:
            user32.ShowWindow(found[0], 9)          # SW_RESTORE
            user32.SetForegroundWindow(found[0])
    except Exception:
        pass


def mintest():
    """端到端验证：窗口最小化后计数是否仍在继续"""
    global CONFIG_PATH, TEMPLATE_PATH, LOG_PATH
    import tempfile
    saved = (CONFIG_PATH, TEMPLATE_PATH, LOG_PATH)
    tmp = tempfile.mkdtemp(prefix='scounter_mintest_')
    CONFIG_PATH = os.path.join(tmp, 'config.json')
    TEMPLATE_PATH = os.path.join(tmp, 'template.png')
    LOG_PATH = os.path.join(tmp, 'counter_log.csv')
    try:
        root = tk.Tk()
        root.geometry('+900+420')          # 远离监测区域
        app = App(root)
        root.update()

        # 用屏幕上一块静态区域自制模板，作为「必然出现」的标识
        region, tpl, score = None, None, 0.0
        for off in (0, 120, 240):
            r = [off, off, 220, 140]
            img = _grab_region(tuple(r))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cand = gray[20:70, 30:100].copy()
            det = Detector(cand, threshold=0.8)
            s, _, _ = det.match_best(cv2.cvtColor(_grab_region(tuple(r)),
                                                  cv2.COLOR_BGR2GRAY))
            if s > score:
                region, tpl, score = r, cand, s
            if score >= 0.9:
                break
        if score < 0.8:
            print('MINT FAIL: 屏幕取样不稳定，score=%.3f' % score)
            return False

        with app.lock:
            app.region = region
            app.tpl_gray = tpl
            app.detector = Detector(tpl, threshold=0.8)
            app._m_mode = 'appear'
            app._m_interval = 0.1
            app._running = True

        root.iconify()                     # 关键：最小化窗口
        time.sleep(2.0)
        with app.lock:
            cnt = app.count
            last_score = app.view.get('score')
        app._stop.set()
        root.destroy()

        ok = cnt >= 1
        print('MINT %s: 最小化期间计数=%s 相似度=%s' % (
            'PASS' if ok else 'FAIL', cnt,
            '%.3f' % last_score if last_score is not None else 'na'))
        return ok
    finally:
        CONFIG_PATH, TEMPLATE_PATH, LOG_PATH = saved


def main():
    root = tk.Tk()
    if not _ensure_single_instance():
        # 已在运行：激活已有窗口后直接退出，不弹阻塞对话框
        root.destroy()
        _activate_existing_window()
        return
    App(root)
    root.mainloop()


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
    elif '--guitest' in sys.argv:
        guitest()
        print('GUITEST PASS')
    elif '--mintest' in sys.argv:
        sys.exit(0 if mintest() else 1)
    elif '--capturetest' in sys.argv:
        sys.exit(0 if capturetest() else 1)
    elif '--tpltest' in sys.argv:
        sys.exit(0 if tpltest() else 1)
    else:
        main()
