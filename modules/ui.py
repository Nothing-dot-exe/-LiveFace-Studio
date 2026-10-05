"""PySide6 UI for LiveFace-Studio.

Public API kept stable for the rest of the codebase:
    init(start, destroy, lang) -> _Window
        Returned object has .mainloop() that core.py calls.
    update_status(text)
        Thread-safe; routed through Qt signal when called off-UI.
    check_and_ignore_nsfw(target, destroy=None) -> bool
"""

from __future__ import annotations

import os
import platform
import queue
import sys
import tempfile
import threading
import time
import webbrowser
from typing import Callable, List, Optional, Tuple

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
from PySide6.QtCore import (
    QObject,
    QThread,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import modules.globals
import modules.metadata
from modules.capturer import get_video_frame, get_video_frame_total
from modules.face_analyser import (
    add_blank_map,
    detect_many_faces_fast,
    detect_one_face_fast,
    ensure_landmarks,
    get_one_face,
    get_unique_faces_from_target_image,
    get_unique_faces_from_target_video,
    has_valid_map,
    reset_face_analyser,
    simplify_maps,
)
from modules.gettext import LanguageManager
from modules.gpu_processing import gpu_cvt_color, gpu_flip, gpu_resize
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    has_image_extension,
    is_image,
    is_video,
)
from modules import imread_unicode
from modules.video_capture import VideoCapturer

if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph

import json


# ─── constants ────────────────────────────────────────────────────────────

ROOT_HEIGHT = 650
ROOT_WIDTH = 490

PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200
PREVIEW_DEFAULT_WIDTH = 640
PREVIEW_DEFAULT_HEIGHT = 360
LIVE_PREVIEW_DEFAULT_WIDTH = 1100
LIVE_PREVIEW_DEFAULT_HEIGHT = 600

POPUP_WIDTH = 750
POPUP_HEIGHT = 810
POPUP_SCROLL_WIDTH = 720
POPUP_SCROLL_HEIGHT = 700

POPUP_LIVE_WIDTH = 900
POPUP_LIVE_HEIGHT = 820
POPUP_LIVE_SCROLL_WIDTH = 870
POPUP_LIVE_SCROLL_HEIGHT = 700

MAPPER_PREVIEW_SIZE = 100
SOURCE_TARGET_PREVIEW_SIZE = 160


# ─── classic dark stylesheet (v1.0.1 style) ───────────────────────────────

QSS = """
QMainWindow, QDialog { background-color: #1a1a1a; color: #e6e6e6; }
QWidget { color: #e6e6e6; font-family: "Segoe UI", Arial, sans-serif; font-size: 11pt; }

QPushButton {
    background-color: #1f538d;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: 500;
    font-size: 11pt;
}
QPushButton:hover  { background-color: #2465ab; }
QPushButton:pressed{ background-color: #174373; }
QPushButton:disabled { background-color: #2e2e2e; color: #666666; }

QComboBox {
    background-color: #242424;
    color: #e6e6e6;
    border: 1px solid #383838;
    border-radius: 4px;
    padding: 4px 10px;
    min-height: 24px;
    font-size: 10pt;
}
QComboBox:hover { border-color: #1f538d; }
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 22px;
    border-left-width: 0px;
}
QComboBox QAbstractItemView {
    background-color: #242424;
    color: #e6e6e6;
    selection-background-color: #1f538d;
    selection-color: #ffffff;
    border: 1px solid #383838;
}

QLabel#imageDrop {
    background-color: #1a1a1a;
    border: none;
}
QLabel#statusLabel {
    color: #888888;
    font-size: 9.5pt;
}
QLabel#linkLabel {
    color: #555555;
    font-size: 9.5pt;
}
QLabel#linkLabel:hover {
    color: #3b82f6;
}

QScrollArea { border: none; background: transparent; }
"""


# ─── module-level state ───────────────────────────────────────────────────

_APP: Optional[QApplication] = None
_MAIN: Optional["MainWindow"] = None
_PREVIEW: Optional["PreviewWindow"] = None
_WEBCAM_PREVIEW: Optional["WebcamPreviewWindow"] = None
_MAPPER: Optional["MapperDialog"] = None
_LIVE_MAPPER: Optional["LiveMapperDialog"] = None
_LANG: Optional[LanguageManager] = None
_BRIDGE: Optional["_UIBridge"] = None


def _(text: str) -> str:
    """Translate via LanguageManager; falls back to identity."""
    if _LANG is None:
        return text
    return _LANG._(text)


# Preserve original cwd state for file dialogs.
_RECENT_SOURCE_DIR: Optional[str] = None
_RECENT_TARGET_DIR: Optional[str] = None
_RECENT_OUTPUT_DIR: Optional[str] = None

# QFileDialog filter strings, built from the canonical extension sets in
# globals so every dialog stays in sync (no hand-copied lists to drift).
_IMAGE_FILE_FILTER = "Images (" + " ".join(
    f"*{ext}" for ext in modules.globals.IMAGE_EXTENSIONS
) + ")"
_MEDIA_FILE_FILTER = "Media (" + " ".join(
    f"*{ext}" for ext in (*modules.globals.IMAGE_EXTENSIONS, *modules.globals.VIDEO_EXTENSIONS)
) + ")"
_VIDEO_FILE_FILTER = "Videos (" + " ".join(
    f"*{ext}" for ext in modules.globals.VIDEO_EXTENSIONS
) + ")"


# ─── image utilities ─────────────────────────────────────────────────────


def fit_image_to_size(image, width: int, height: int):
    """BGR ndarray → BGR ndarray scaled to fit within (width, height)."""
    if width is None and height is None or width <= 0 or height <= 0:
        return image
    h, w = image.shape[:2]
    ratio_w = width / w
    ratio_h = height / h
    ratio = min(ratio_w, ratio_h)
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    return gpu_resize(image, dsize=new_size)


def _bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """Zero-copy BGR ndarray → QPixmap."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def _pil_to_qpixmap(image: Image.Image) -> QPixmap:
    """PIL.Image → QPixmap."""
    image = image.convert("RGBA")
    data = image.tobytes("raw", "RGBA")
    qimg = QImage(data, image.width, image.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


def get_face_display_name(source_path: str | None) -> str:
    """Get a clean, human-readable name for the target face image."""
    if not source_path:
        return "Target Face"
    fname = os.path.basename(source_path)
    FRIENDLY_NAMES = {
        "baba.png": "Baba",
        "elon.webp": "Elon Musk",
        "elon.png": "Elon Musk",
        "India_PM_Narendra_Modi-2.jpg": "Narendra Modi",
        "narendra_modi.jpg": "Narendra Modi",
        "trump.jpg": "Donald Trump",
        "putin.webp": "Vladimir Putin",
        "rdj.jpg": "Robert Downey Jr.",
        "rahul gandhi.jpg": "Rahul Gandhi",
        "jonsnow.jpg": "Jon Snow",
        "jonsnow.avif": "Jon Snow",
        "tyrion.jpg": "Tyrion Lannister",
        "stive.jpg": "Steve Jobs",
        "Adi.png": "Adi",
        "Screenshot 2026-10-04 205420.png": "User",
        "screenshot_user.png": "User",
    }
    if fname in FRIENDLY_NAMES:
        return FRIENDLY_NAMES[fname]
    name_no_ext = os.path.splitext(fname)[0]
    clean = name_no_ext.replace("_", " ").replace("-", " ").strip()
    return clean.title() if clean else "Target Face"


def get_available_faces() -> List[Tuple[str, str]]:
    """Returns a list of (display_name, file_path) sorted by display name."""
    faces = []
    seen_paths = set()
    seen_names = set()
    search_dirs = [
        os.path.normpath(os.path.join(modules.globals.ROOT_DIR, "..", "media", "faces")),
        r"C:\Users\kadam\Downloads\images",
        os.path.normpath(os.path.join(modules.globals.ROOT_DIR, "..", "media")),
    ]
    for sdir in search_dirs:
        if not os.path.isdir(sdir):
            continue
        try:
            for fname in sorted(os.listdir(sdir)):
                if fname.lower() in ("instruction.png", "download.png", "screenshot_user.png"):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext not in modules.globals.IMAGE_EXTENSIONS and ext != ".avif":
                    continue
                fpath = os.path.normpath(os.path.join(sdir, fname))
                if fpath in seen_paths:
                    continue
                disp_name = get_face_display_name(fpath)
                if disp_name in seen_names or disp_name == "Instruction":
                    continue
                seen_paths.add(fpath)
                seen_names.add(disp_name)
                faces.append((disp_name, fpath))
        except Exception:
            pass
    faces.sort(key=lambda x: x[0])
    return faces


def draw_face_overlay(frame: np.ndarray, face_obj, label_text: str, color=(0, 255, 0)) -> None:
    """Draws a crisp green bounding box with sleek corner accents and a high-contrast label pill."""
    if not hasattr(face_obj, "bbox") or face_obj.bbox is None:
        return
    bx1, by1, bx2, by2 = face_obj.bbox.astype(int)
    h_f, w_f = frame.shape[:2]
    bx1, by1 = max(0, min(bx1, w_f - 1)), max(0, min(by1, h_f - 1))
    bx2, by2 = max(0, min(bx2, w_f - 1)), max(0, min(by2, h_f - 1))
    if bx2 <= bx1 or by2 <= by1:
        return

    # 1. Main green bounding box rectangle
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)

    # 2. Sleek HUD corner brackets for professional visual feedback
    corner_len = min(22, (bx2 - bx1) // 4, (by2 - by1) // 4)
    if corner_len > 4:
        # Top-left corner
        cv2.line(frame, (bx1, by1), (bx1 + corner_len, by1), color, 3)
        cv2.line(frame, (bx1, by1), (bx1, by1 + corner_len), color, 3)
        # Top-right corner
        cv2.line(frame, (bx2, by1), (bx2 - corner_len, by1), color, 3)
        cv2.line(frame, (bx2, by1), (bx2, by1 + corner_len), color, 3)
        # Bottom-left corner
        cv2.line(frame, (bx1, by2), (bx1 + corner_len, by2), color, 3)
        cv2.line(frame, (bx1, by2), (bx1, by2 - corner_len), color, 3)
        # Bottom-right corner
        cv2.line(frame, (bx2, by2), (bx2 - corner_len, by2), color, 3)
        cv2.line(frame, (bx2, by2), (bx2, by2 - corner_len), color, 3)

    # 3. High-contrast label pill badge above top edge
    if label_text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.70
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        pad_x = 8
        pad_y = 5
        pill_w = tw + pad_x * 2
        pill_h = th + pad_y * 2 + baseline

        # Position centered above top line if there's room, otherwise inside the box at top
        px1 = max(2, min(bx1 + (bx2 - bx1 - pill_w) // 2, w_f - pill_w - 2))
        if by1 - pill_h - 4 >= 2:
            py1 = by1 - pill_h - 4
        else:
            py1 = by1 + 4
        px2 = px1 + pill_w
        py2 = py1 + pill_h

        # Solid dark badge background with green outline
        cv2.rectangle(frame, (px1, py1), (px2, py2), (15, 15, 15), -1)
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 1)

        # Crisp text inside pill
        text_x = px1 + pad_x
        text_y = py1 + pad_y + th
        cv2.putText(
            frame,
            label_text,
            (text_x, text_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def render_image_preview(image_path: str, size: Tuple[int, int]) -> QPixmap:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return _pil_to_qpixmap(image)


def render_video_preview(
    video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> Optional[QPixmap]:
    capture = cv2.VideoCapture(video_path)
    try:
        if frame_number:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        has_frame, frame = capture.read()
        if not has_frame:
            return None
        image = Image.fromarray(gpu_cvt_color(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return _pil_to_qpixmap(image)
    finally:
        capture.release()


# ─── persistence ─────────────────────────────────────────────────────────


def save_switch_states():
    state = {
        "keep_fps": modules.globals.keep_fps,
        "keep_audio": modules.globals.keep_audio,
        "keep_frames": modules.globals.keep_frames,
        "many_faces": modules.globals.many_faces,
        "map_faces": modules.globals.map_faces,
        "poisson_blend": modules.globals.poisson_blend,
        "color_correction": modules.globals.color_correction,
        "nsfw_filter": modules.globals.nsfw_filter,
        "live_mirror": modules.globals.live_mirror,
        "live_resizable": modules.globals.live_resizable,
        "fp_ui": modules.globals.fp_ui,
        "show_fps": modules.globals.show_fps,
        "mouth_mask": modules.globals.mouth_mask,
        "show_mouth_mask_box": modules.globals.show_mouth_mask_box,
        "mouth_mask_size": modules.globals.mouth_mask_size,
        "capture_resolution": list(modules.globals.capture_resolution),
        "det_size": modules.globals.det_size,
    }
    try:
        with open("switch_states.json", "w") as f:
            json.dump(state, f)
    except OSError:
        pass


def load_switch_states():
    try:
        with open("switch_states.json", "r") as f:
            state = json.load(f)
        modules.globals.keep_fps = state.get("keep_fps", True)
        modules.globals.keep_audio = state.get("keep_audio", True)
        modules.globals.keep_frames = state.get("keep_frames", False)
        modules.globals.many_faces = state.get("many_faces", False)
        modules.globals.map_faces = state.get("map_faces", False)
        modules.globals.poisson_blend = state.get("poisson_blend", False)
        modules.globals.color_correction = state.get("color_correction", False)
        modules.globals.nsfw_filter = state.get("nsfw_filter", False)
        modules.globals.live_mirror = state.get("live_mirror", False)
        modules.globals.live_resizable = state.get("live_resizable", False)
        modules.globals.fp_ui = state.get("fp_ui", {"face_enhancer": False})
        modules.globals.show_fps = state.get("show_fps", False)
        # Mouth mask always starts disabled (slider at 0) on launch,
        # regardless of the persisted value — enable it explicitly each session.
        modules.globals.mouth_mask_size = 0.0
        modules.globals.mouth_mask = False
        modules.globals.show_mouth_mask_box = False
        # Validate persisted camera settings before trusting them — a hand-
        # edited/corrupt file shouldn't push a bad size into cv2/insightface.
        res = state.get("capture_resolution")
        if isinstance(res, (list, tuple)) and len(res) == 2:
            try:
                w, h = int(res[0]), int(res[1])
                if w > 0 and h > 0:
                    modules.globals.capture_resolution = (w, h)
            except (TypeError, ValueError):
                pass
        if state.get("det_size") in (160, 320, 640):
            modules.globals.det_size = int(state["det_size"])
    except FileNotFoundError:
        pass
    except (OSError, json.JSONDecodeError):
        pass


# ─── thread-safe status bridge ───────────────────────────────────────────


class _UIBridge(QObject):
    """Single QObject that owns cross-thread signals."""

    statusChanged = Signal(str)


def _emit_status(text: str) -> None:
    if _BRIDGE is None:
        print(text)
        return
    _BRIDGE.statusChanged.emit(text)


# ─── public API ──────────────────────────────────────────────────────────


def update_status(text: str) -> None:
    """Thread-safe status update — uses signal if called off-UI thread."""
    _emit_status(_(text))
    if _APP is not None and QThread.currentThread() is _APP.thread():
        # On UI thread — flush events so the user sees the update during
        # long synchronous start() runs.
        _APP.processEvents()


def check_and_ignore_nsfw(target, destroy: Optional[Callable] = None) -> bool:
    from numpy import ndarray
    from modules.predicter import predict_frame, predict_image, predict_video

    check_nsfw = None
    if isinstance(target, str):
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif isinstance(target, ndarray):
        check_nsfw = predict_frame

    if check_nsfw and check_nsfw(target):
        if destroy:
            destroy(to_quit=False)
        update_status("Processing ignored!")
        return True
    return False


# ─── camera enumeration (unchanged from tk version) ──────────────────────


def get_available_cameras() -> Tuple[List[int], List[str]]:
    if platform.system() == "Windows":
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            if devices:
                return list(range(len(devices))), devices
            return [], ["No cameras found"]
        except Exception as exc:
            print(f"Error detecting cameras: {exc}")
            return [], ["No cameras found"]

    if platform.system() == "Darwin":
        return [0, 1], ["Camera 0", "Camera 1"]

    # Linux probe
    indices: List[int] = []
    names: List[str] = []
    for i in range(10):
        cap = cv2.VideoCapture(f"/dev/video{i}")
        if cap.isOpened():
            indices.append(i)
            names.append(f"Camera {i}")
            cap.release()
    return (indices, names) if names else ([], ["No cameras found"])


# ─── main window ─────────────────────────────────────────────────────────


def _make_image_drop(text: str, size: Tuple[int, int]) -> QLabel:
    label = QLabel(text)
    label.setObjectName("imageDrop")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setFixedSize(size[0], size[1])
    label.setText(text)
    return label


class QSwitch(QWidget):
    """Custom toggle switch matching CustomTkinter pill style."""

    toggled = Signal(bool)

    def __init__(self, text: str = "", checked: bool = False, tooltip: str = "", parent=None):
        super().__init__(parent)
        self._checked = checked
        self._text = text
        if tooltip:
            self.setToolTip(tooltip)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(30)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, value: bool) -> None:
        if self._checked != value:
            self._checked = value
            self.update()
            self.toggled.emit(self._checked)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._checked = not self._checked
            self.update()
            self.toggled.emit(self._checked)
        super().mousePressEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Track dimensions
        track_w = 42
        track_h = 22
        track_x = 2
        track_y = (self.height() - track_h) // 2

        track_color = QColor("#1f538d") if self._checked else QColor("#3c3c3c")
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(track_color))
        p.drawRoundedRect(track_x, track_y, track_w, track_h, track_h / 2, track_h / 2)

        # Thumb
        thumb_dia = 16
        thumb_y = track_y + (track_h - thumb_dia) // 2
        thumb_x = track_x + track_w - thumb_dia - 3 if self._checked else track_x + 3

        thumb_color = QColor("#ffffff") if self._checked else QColor("#cfcfcf")
        p.setBrush(QBrush(thumb_color))
        p.drawEllipse(thumb_x, thumb_y, thumb_dia, thumb_dia)

        # Label text
        if self._text:
            p.setPen(QColor("#e6e6e6"))
            font = p.font()
            font.setPointSize(11)
            font.setFamily("Segoe UI")
            p.setFont(font)
            text_x = track_x + track_w + 12
            p.drawText(
                text_x, 0, self.width() - text_x, self.height(),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                self._text,
            )
        p.end()


_Switch = QSwitch


class MainWindow(QMainWindow):
    def __init__(self, start_cb: Callable, destroy_cb: Callable):
        super().__init__()
        load_switch_states()
        self._start_cb = start_cb
        self._destroy_cb = destroy_cb

        self.setWindowTitle(f"{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}")
        self.setFixedSize(ROOT_WIDTH, ROOT_HEIGHT)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(28, 18, 28, 14)
        layout.setSpacing(10)

        # 1. Source / Target image preview + selection buttons
        layout.addLayout(self._build_image_row())

        layout.addSpacing(4)

        # 2. Options Switches (2 columns of 3 switches matching instruction.png)
        layout.addLayout(self._build_options_row())

        layout.addSpacing(4)

        # 3. Action Buttons (Start, Destroy, Preview / Live)
        layout.addLayout(self._build_action_grid())

        layout.addSpacing(2)

        # 4. Camera Selection (compact, auto-selects Iriun Webcam)
        layout.addLayout(self._build_camera_row())

        # 5. Status message
        self._status_label = QLabel("")
        self._status_label.setObjectName("statusLabel")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setFixedHeight(22)
        layout.addWidget(self._status_label)

        # 6. Footer link
        footer = QLabel("Deep Live Cam")
        footer.setObjectName("linkLabel")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setCursor(Qt.CursorShape.PointingHandCursor)
        footer.mousePressEvent = lambda _e: webbrowser.open("https://deeplivecam.net")
        layout.addWidget(footer)

        # 7. Select initial preset face on startup if available and none selected yet
        if self._available_faces and not modules.globals.source_path:
            baba_idx = next(
                (i for i, (dn, _) in enumerate(self._available_faces) if "baba" in dn.lower()),
                0,
            )
            self.combo_source_face.blockSignals(True)
            self.combo_source_face.setCurrentIndex(baba_idx)
            self.combo_source_face.blockSignals(False)
            self._set_active_source_path(self._available_faces[baba_idx][1])

    # ── image row ────────────────────────────────────────────────────────

    def _build_image_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(28)
        row.setContentsMargins(0, 0, 0, 0)

        # Source column
        src_col = QVBoxLayout()
        src_col.setSpacing(5)
        src_col.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.source_label = QLabel()
        self.source_label.setObjectName("imageDrop")
        self.source_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.source_label.setFixedSize(160, 160)
        self.source_label.setStyleSheet("border: 1px dashed #3a3a3a; border-radius: 6px; background-color: #141414;")
        src_col.addWidget(self.source_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Preset Face Dropdown
        self.combo_source_face = QComboBox()
        self.combo_source_face.setFixedSize(160, 28)
        self.combo_source_face.setToolTip("Select preset face (Baba, Bhauji, Elon, etc.)")
        self._available_faces = get_available_faces()
        for dname, fpath in self._available_faces:
            self.combo_source_face.addItem(dname, fpath)
        self.combo_source_face.currentIndexChanged.connect(self._on_preset_face_selected)
        src_col.addWidget(self.combo_source_face, alignment=Qt.AlignmentFlag.AlignCenter)

        # Prev / Next Face Buttons row
        nav_row = QHBoxLayout()
        nav_row.setSpacing(6)
        nav_row.setContentsMargins(0, 0, 0, 0)

        self.btn_face_prev = QPushButton("◀ Prev")
        self.btn_face_prev.setFixedSize(77, 28)
        self.btn_face_prev.setToolTip("Previous face")
        self.btn_face_prev.setStyleSheet("QPushButton { font-size: 9pt; padding: 2px; }")
        self.btn_face_prev.clicked.connect(self._on_prev_face)
        nav_row.addWidget(self.btn_face_prev)

        self.btn_face_next = QPushButton("Next ▶")
        self.btn_face_next.setFixedSize(77, 28)
        self.btn_face_next.setToolTip("Next face")
        self.btn_face_next.setStyleSheet("QPushButton { font-size: 9pt; padding: 2px; }")
        self.btn_face_next.clicked.connect(self._on_next_face)
        nav_row.addWidget(self.btn_face_next)

        src_col.addLayout(nav_row)

        self.btn_select_source = QPushButton(_("Select a face"))
        self.btn_select_source.setFixedSize(160, 30)
        self.btn_select_source.setToolTip(_("Choose a custom source face image from your computer"))
        self.btn_select_source.clicked.connect(self._on_select_source)
        src_col.addWidget(self.btn_select_source, alignment=Qt.AlignmentFlag.AlignCenter)

        # Target column
        tgt_col = QVBoxLayout()
        tgt_col.setSpacing(5)
        tgt_col.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.target_label = QLabel()
        self.target_label.setObjectName("imageDrop")
        self.target_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.target_label.setFixedSize(160, 160)
        self.target_label.setStyleSheet("border: 1px dashed #3a3a3a; border-radius: 6px; background-color: #141414;")
        tgt_col.addWidget(self.target_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.lbl_target_info = QLabel("Target Image / Video")
        self.lbl_target_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_target_info.setStyleSheet("color: #777777; font-size: 9pt; min-height: 28px;")
        tgt_col.addWidget(self.lbl_target_info, alignment=Qt.AlignmentFlag.AlignCenter)

        tgt_spacer = QLabel("")
        tgt_spacer.setFixedHeight(28)
        tgt_col.addWidget(tgt_spacer, alignment=Qt.AlignmentFlag.AlignCenter)

        self.btn_select_target = QPushButton(_("Select a target"))
        self.btn_select_target.setFixedSize(160, 30)
        self.btn_select_target.setToolTip(_("Choose the target image or video to apply face swap to"))
        self.btn_select_target.clicked.connect(self._on_select_target)
        tgt_col.addWidget(self.btn_select_target, alignment=Qt.AlignmentFlag.AlignCenter)

        row.addLayout(src_col)
        row.addLayout(tgt_col)
        return row

    # ── options row (matches instruction.png) ───────────────────────────

    def _build_options_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(36)
        row.setContentsMargins(12, 0, 12, 0)

        # Left column: Keep fps, Keep frames, Face Enhancer
        col_left = QVBoxLayout()
        col_left.setSpacing(8)

        self.sw_keep_fps = QSwitch(
            _("Keep fps"), modules.globals.keep_fps,
            _("Output video keeps the original frame rate")
        )
        self.sw_keep_fps.toggled.connect(
            lambda v: self._set_global("keep_fps", v)
        )

        self.sw_keep_frames = QSwitch(
            _("Keep frames"), modules.globals.keep_frames,
            _("Keep extracted frames on disk after processing")
        )
        self.sw_keep_frames.toggled.connect(
            lambda v: self._set_global("keep_frames", v)
        )

        enhancer_active = modules.globals.fp_ui.get("face_enhancer", False) or getattr(modules.globals, "live_enhancer_enabled", False)
        self.sw_face_enhancer = QSwitch(
            _("Face Enhancer"), enhancer_active,
            _("Enhance face quality (Fast GPEN-256 for smooth live or GFPGAN)")
        )
        self.sw_face_enhancer.toggled.connect(
            lambda v: _update_tumbler("face_enhancer", v)
        )

        self.combo_enhancer_model = QComboBox()
        self.combo_enhancer_model.setStyleSheet(
            "QComboBox { background-color: #242424; color: #ffffff; border: 1px solid #383838; "
            "border-radius: 4px; padding: 2px 6px; font-size: 8.5pt; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { background-color: #242424; color: #ffffff; selection-background-color: #1f538d; }"
        )
        self.combo_enhancer_model.addItem("⚡ Fast (GPEN-256) — Ultra-Smooth Live", "gpen256")
        self.combo_enhancer_model.addItem("🎯 Balanced (GPEN-512) — Crisp Live", "gpen512")
        self.combo_enhancer_model.addItem("💎 Quality (GFPGAN-1024) — Max Details", "gfpgan")
        cur_model = getattr(modules.globals, "live_enhancer_model", "gpen256")
        idx = {"gpen256": 0, "gpen512": 1, "gfpgan": 2}.get(cur_model, 0)
        self.combo_enhancer_model.setCurrentIndex(idx)
        self.combo_enhancer_model.currentIndexChanged.connect(self._on_enhancer_model_changed)

        col_left.addWidget(self.sw_keep_fps)
        col_left.addWidget(self.sw_keep_frames)
        col_left.addWidget(self.sw_face_enhancer)
        col_left.addWidget(self.combo_enhancer_model)

        # Right column: Keep audio, Many faces, NSFW
        col_right = QVBoxLayout()
        col_right.setSpacing(8)

        self.sw_keep_audio = QSwitch(
            _("Keep audio"), modules.globals.keep_audio,
            _("Copy audio track from the source video to output")
        )
        self.sw_keep_audio.toggled.connect(
            lambda v: self._set_global("keep_audio", v)
        )

        self.sw_many_faces = QSwitch(
            _("Many faces"), modules.globals.many_faces,
            _("Swap every detected face, not just the primary one")
        )
        self.sw_many_faces.toggled.connect(
            lambda v: self._set_global("many_faces", v)
        )

        self.sw_nsfw = QSwitch(
            _("NSFW"), modules.globals.nsfw_filter,
            _("Filter NSFW content")
        )
        self.sw_nsfw.toggled.connect(
            lambda v: self._set_global("nsfw_filter", v)
        )

        col_right.addWidget(self.sw_keep_audio)
        col_right.addWidget(self.sw_many_faces)
        col_right.addWidget(self.sw_nsfw)

        row.addLayout(col_left)
        row.addLayout(col_right)
        return row

    def _set_global(self, field: str, value: bool) -> None:
        setattr(modules.globals, field, value)
        save_switch_states()

    def _on_enhancer_model_changed(self, idx: int) -> None:
        model_key = self.combo_enhancer_model.itemData(idx)
        modules.globals.live_enhancer_model = model_key
        print(f"[UI] Enhancer model set to: {model_key}")

    # ── action grid (Start, Destroy, Preview, and centered Live) ─────────

    def _build_action_grid(self) -> QVBoxLayout:
        wrapper = QVBoxLayout()
        wrapper.setSpacing(8)
        wrapper.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Top row: Start, Destroy, Preview
        row_top = QHBoxLayout()
        row_top.setSpacing(16)
        row_top.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btn_start = QPushButton(_("Start"))
        self.btn_start.setFixedSize(110, 36)
        self.btn_start.setToolTip(_("Begin processing the target image/video with selected face"))
        self.btn_start.clicked.connect(self._on_start)

        self.btn_destroy = QPushButton(_("Destroy"))
        self.btn_destroy.setFixedSize(110, 36)
        self.btn_destroy.setToolTip(_("Stop processing and close the application"))
        self.btn_destroy.clicked.connect(lambda: self._destroy_cb())

        self.btn_preview = QPushButton(_("Preview"))
        self.btn_preview.setFixedSize(110, 36)
        self.btn_preview.setToolTip(_("Show/hide a preview of the processed output"))
        self.btn_preview.clicked.connect(self._on_toggle_preview)

        row_top.addWidget(self.btn_start)
        row_top.addWidget(self.btn_destroy)
        row_top.addWidget(self.btn_preview)

        # Bottom row: Live (centered directly below Destroy)
        row_bottom = QHBoxLayout()
        row_bottom.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btn_live = QPushButton(_("Live"))
        self.btn_live.setFixedSize(110, 36)
        self.btn_live.setToolTip(_("Start real-time face swap using webcam"))
        self.btn_live.clicked.connect(self._on_live)

        row_bottom.addWidget(self.btn_live)

        wrapper.addLayout(row_top)
        wrapper.addLayout(row_bottom)
        return wrapper

    # ── camera row ───────────────────────────────────────────────────────

    def _build_camera_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.setSpacing(10)

        self._camera_indices, self._camera_names = get_available_cameras()
        self.cb_camera = QComboBox()
        self.cb_camera.setFixedWidth(240)
        self.cb_camera.setFixedHeight(30)

        if not self._camera_names or self._camera_names[0] == "No cameras found":
            self.cb_camera.addItem("No cameras found")
            self.cb_camera.setEnabled(False)
            self.btn_live.setEnabled(False)
        else:
            self.cb_camera.addItems(self._camera_names)
            # Auto-select Iriun Webcam if detected
            for idx, name in enumerate(self._camera_names):
                if "iriun" in name.lower():
                    self.cb_camera.setCurrentIndex(idx)
                    break

        self.cb_camera.setToolTip(_("Select which camera to use for live mode"))
        row.addWidget(self.cb_camera)
        return row

    # ── slot handlers ────────────────────────────────────────────────────

    def set_status(self, text: str) -> None:
        if hasattr(self, "_status_label") and self._status_label is not None:
            self._status_label.setText(text)

    def _on_preset_face_selected(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._available_faces):
            return
        _, fpath = self._available_faces[idx]
        self._set_active_source_path(fpath)

    def _on_prev_face(self) -> None:
        if not self._available_faces:
            return
        idx = (self.combo_source_face.currentIndex() - 1) % len(self._available_faces)
        self.combo_source_face.setCurrentIndex(idx)

    def _on_next_face(self) -> None:
        if not self._available_faces:
            return
        idx = (self.combo_source_face.currentIndex() + 1) % len(self._available_faces)
        self.combo_source_face.setCurrentIndex(idx)

    def _set_active_source_path(self, fpath: str) -> None:
        norm_path = os.path.normpath(fpath)
        modules.globals.source_path = norm_path
        self.source_label.setPixmap(render_image_preview(norm_path, (160, 160)))
        self.source_label.setText("")
        disp_name = get_face_display_name(norm_path)
        self.set_status(f"Active face: {disp_name}")
        # Sync with WebcamPreviewWindow if open
        if _WEBCAM_PREVIEW is not None and hasattr(_WEBCAM_PREVIEW, "_apply_source_path_external"):
            _WEBCAM_PREVIEW._apply_source_path_external(norm_path)

    def sync_source_face(self, fpath: str) -> None:
        """Called when WebcamPreviewWindow changes the active face."""
        norm_path = os.path.normpath(fpath)
        modules.globals.source_path = norm_path
        if hasattr(self, "combo_source_face"):
            for i in range(self.combo_source_face.count()):
                if os.path.normpath(self.combo_source_face.itemData(i)) == norm_path:
                    self.combo_source_face.blockSignals(True)
                    self.combo_source_face.setCurrentIndex(i)
                    self.combo_source_face.blockSignals(False)
                    break
        self.source_label.setPixmap(render_image_preview(norm_path, (160, 160)))
        self.source_label.setText("")
        disp_name = get_face_display_name(norm_path)
        self.set_status(f"Active face: {disp_name}")

    def _on_select_source(self) -> None:
        global _RECENT_SOURCE_DIR
        if _PREVIEW is not None:
            _PREVIEW.hide()
        path, _filter = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if path and is_image(path):
            norm_path = os.path.normpath(path)
            modules.globals.source_path = norm_path
            _RECENT_SOURCE_DIR = os.path.dirname(norm_path)
            self.source_label.setPixmap(render_image_preview(norm_path, (160, 160)))
            self.source_label.setText("")
            # Add to combo if not present
            found = False
            for i in range(self.combo_source_face.count()):
                if os.path.normpath(self.combo_source_face.itemData(i)) == norm_path:
                    self.combo_source_face.blockSignals(True)
                    self.combo_source_face.setCurrentIndex(i)
                    self.combo_source_face.blockSignals(False)
                    found = True
                    break
            if not found:
                dname = get_face_display_name(norm_path)
                self._available_faces.append((dname, norm_path))
                self.combo_source_face.addItem(dname, norm_path)
                self.combo_source_face.blockSignals(True)
                self.combo_source_face.setCurrentIndex(self.combo_source_face.count() - 1)
                self.combo_source_face.blockSignals(False)
            if _WEBCAM_PREVIEW is not None and hasattr(_WEBCAM_PREVIEW, "_apply_source_path_external"):
                _WEBCAM_PREVIEW._apply_source_path_external(norm_path)
            self.set_status(_("Checking face detection..."))
            threading.Thread(target=self._validate_source_face, args=(norm_path,), daemon=True).start()
        elif not path:
            return
        else:
            modules.globals.source_path = None
            self.source_label.clear()
            self.source_label.setText("")
            self.set_status("")

    def _validate_source_face(self, path: str) -> None:
        try:
            cv2_img = imread_unicode(path)
            if cv2_img is not None:
                face = get_one_face(cv2_img)
                if face is None:
                    _emit_status(_("⚠️ No face detected in this image! Please choose a clear front photo."))
                else:
                    _emit_status(_("✅ Face detected successfully! Ready for Live."))
        except Exception as exc:
            print(f"[Face Check Error]: {exc}")

    def _on_select_target(self) -> None:
        global _RECENT_TARGET_DIR
        if _PREVIEW is not None:
            _PREVIEW.hide()
        path, _filter = QFileDialog.getOpenFileName(
            self, _("select an target image or video"),
            _RECENT_TARGET_DIR or "",
            _MEDIA_FILE_FILTER,
        )
        if not path:
            return
        if is_image(path):
            modules.globals.target_path = path
            _RECENT_TARGET_DIR = os.path.dirname(path)
            self.target_label.setPixmap(render_image_preview(path, (160, 160)))
            self.target_label.setText("")
            self.set_status(_("Target image loaded"))
        elif is_video(path):
            modules.globals.target_path = path
            _RECENT_TARGET_DIR = os.path.dirname(path)
            pm = render_video_preview(path, (160, 160))
            if pm:
                self.target_label.setPixmap(pm)
                self.target_label.setText("")
            self.set_status(_("Target video loaded"))
        else:
            modules.globals.target_path = None
            self.target_label.clear()
            self.target_label.setText("")

    def _on_start(self) -> None:
        if _MAPPER is not None and _MAPPER.isVisible():
            update_status("Please complete pop-up or close it.")
            return
        if modules.globals.map_faces:
            modules.globals.source_target_map = []
            if is_image(modules.globals.target_path):
                update_status("Getting unique faces")
                get_unique_faces_from_target_image()
            elif is_video(modules.globals.target_path):
                update_status("Getting unique faces")
                get_unique_faces_from_target_video()
            if modules.globals.source_target_map:
                _open_mapper_dialog(self._start_cb, modules.globals.source_target_map)
            else:
                update_status("No faces found in target")
        else:
            self._select_output_and_start()

    def _select_output_and_start(self) -> None:
        global _RECENT_OUTPUT_DIR
        if is_image(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self, _("save image output file"),
                os.path.join(_RECENT_OUTPUT_DIR or "", "output.png"),
                _IMAGE_FILE_FILTER,
            )
        elif is_video(modules.globals.target_path):
            path, _f = QFileDialog.getSaveFileName(
                self, _("save video output file"),
                os.path.join(_RECENT_OUTPUT_DIR or "", "output.mp4"),
                _VIDEO_FILE_FILTER,
            )
        else:
            return
        if path:
            modules.globals.output_path = path
            _RECENT_OUTPUT_DIR = os.path.dirname(path)
            self._start_cb()

    def _on_toggle_preview(self) -> None:
        if _PREVIEW is None:
            return
        if _PREVIEW.isVisible():
            _PREVIEW.hide()
        elif modules.globals.source_path and modules.globals.target_path:
            _PREVIEW.init_for_target()
            _PREVIEW.refresh_frame(0)
            _PREVIEW.show()

    def _on_live(self) -> None:
        idx = self.cb_camera.currentIndex()
        if idx < 0 or idx >= len(self._camera_indices):
            update_status("No camera available")
            return
        camera_index = self._camera_indices[idx]
        if _LIVE_MAPPER is not None and _LIVE_MAPPER.isVisible():
            update_status("Source x Target Mapper is already open.")
            _LIVE_MAPPER.raise_()
            return
        if modules.globals.source_path is None:
            if hasattr(self, "_available_faces") and self._available_faces:
                fpath = self._available_faces[0][1]
                modules.globals.source_path = fpath
                if hasattr(self, "combo_source_face"):
                    self.combo_source_face.blockSignals(True)
                    self.combo_source_face.setCurrentIndex(0)
                    self.combo_source_face.blockSignals(False)
            else:
                update_status("Please select a source image first")
                return
        src_img = imread_unicode(modules.globals.source_path)
        if src_img is None:
            update_status("⚠️ Could not load source image! Try another picture.")
            return
        from modules.face_analyser import get_face_analyser
        from modules.processors.frame.face_swapper import get_face_swapper
        get_face_analyser()
        get_face_swapper()
        _open_webcam_preview(camera_index)

    def closeEvent(self, event):
        # Treat OS-level close as Destroy click
        self._destroy_cb()
        event.accept()


def _update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value
    if var == "face_enhancer":
        modules.globals.live_enhancer_enabled = value
    save_switch_states()
    # If we're currently in a live preview, refresh frame processors so
    # toggling enhancers takes effect immediately.
    if _WEBCAM_PREVIEW is not None and _WEBCAM_PREVIEW.isVisible():
        get_frame_processors_modules(modules.globals.frame_processors)


# ─── preview window (still-image / video scrub) ──────────────────────────


class PreviewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(_("Preview"))
        self.resize(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._image_label, 1)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self.refresh_frame)
        layout.addWidget(self._slider)

    def init_for_target(self) -> None:
        if is_image(modules.globals.target_path):
            self._slider.hide()
        elif is_video(modules.globals.target_path):
            total = get_video_frame_total(modules.globals.target_path)
            self._slider.setRange(0, max(0, total - 1))
            self._slider.setValue(0)
            self._slider.show()

    def refresh_frame(self, frame_number: int = 0) -> None:
        if not (modules.globals.source_path and modules.globals.target_path):
            return
        update_status("Processing...")
        temp_frame = get_video_frame(modules.globals.target_path, frame_number)
        if modules.globals.nsfw_filter and check_and_ignore_nsfw(temp_frame):
            return
        from modules.processors.frame.core import get_frame_processors_modules as _gfpm
        for fp in _gfpm(modules.globals.frame_processors):
            temp_frame = fp.process_frame(
                get_one_face(imread_unicode(modules.globals.source_path)), temp_frame
            )
        # Fit to current widget size while preserving aspect ratio.
        h, w = temp_frame.shape[:2]
        bound_w = min(PREVIEW_MAX_WIDTH, max(self.width(), PREVIEW_DEFAULT_WIDTH))
        bound_h = min(PREVIEW_MAX_HEIGHT, max(self.height(), PREVIEW_DEFAULT_HEIGHT))
        ratio = min(bound_w / w, bound_h / h)
        new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
        temp_frame = cv2.resize(temp_frame, new_size, interpolation=cv2.INTER_LANCZOS4)
        self._image_label.setPixmap(_bgr_to_qpixmap(temp_frame))
        update_status("Processing succeed!")


# ─── webcam preview window ───────────────────────────────────────────────


class _CaptureWorker(QThread):
    """Reads frames from the camera into a bounded queue. Drops on overflow."""

    def __init__(self, cap, capture_queue: queue.Queue, stop_event: threading.Event):
        super().__init__()
        self._cap = cap
        self._queue = capture_queue
        self._stop = stop_event

    def run(self) -> None:
        try:
            while not self._stop.is_set():
                ret, frame = self._cap.read()
                if not ret:
                    self._stop.set()
                    break
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._queue.put_nowait(frame)
                    except queue.Full:
                        pass
        except Exception as exc:
            import traceback
            print(f"[DLC Live _CaptureWorker Error]: {exc}")
            traceback.print_exc()


class _ProcessingWorker(QThread):
    """Pulls raw frames, runs detect/swap/enhance, pushes processed frames."""

    def __init__(self, capture_queue, processed_queue, stop_event, camera_fps: float):
        super().__init__()
        self._cq = capture_queue
        self._pq = processed_queue
        self._stop = stop_event
        self._fps = camera_fps

    def run(self) -> None:
        frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
        source_image = None
        last_source_path = None
        prev_time = time.time()
        fps_update_interval = 0.5
        frame_count = 0
        fps = 0.0
        det_count = 0
        cached_target_face = None
        cached_many_faces = None
        last_valid_faces = None
        det_interval = max(1, round(self._fps * 0.08))

        while not self._stop.is_set():
            try:
                frame = self._cq.get(timeout=0.05)
            except queue.Empty:
                continue

            temp_frame = frame
            if modules.globals.live_mirror:
                temp_frame = gpu_flip(temp_frame, 1)

            orig_frame = temp_frame.copy()

            if not modules.globals.map_faces:
                if (
                    modules.globals.source_path
                    and modules.globals.source_path != last_source_path
                ):
                    last_source_path = modules.globals.source_path
                    source_image = get_one_face(imread_unicode(modules.globals.source_path))
                    if source_image is not None:
                        print(f"[DLC Live] [OK] Source face successfully loaded from: {modules.globals.source_path}")
                    else:
                        print(f"[DLC Live] [WARN] Could not detect face in source photo: {modules.globals.source_path}")

                det_count += 1
                if det_count % det_interval == 0:
                    if modules.globals.many_faces:
                        cached_target_face = None
                        cached_many_faces = detect_many_faces_fast(temp_frame)
                    else:
                        cached_target_face = detect_one_face_fast(temp_frame)
                        cached_many_faces = None

                cached_faces = None
                if cached_many_faces:
                    cached_faces = cached_many_faces
                    last_valid_faces = cached_many_faces
                elif cached_target_face is not None:
                    cached_faces = [cached_target_face]
                    last_valid_faces = [cached_target_face]
                elif last_valid_faces is not None:
                    cached_faces = last_valid_faces

                if modules.globals.mouth_mask and cached_faces:
                    ensure_landmarks(temp_frame, cached_faces)

                # 1. Face Swapper
                swapper_mod = next((m for m in frame_processors if m.NAME == "DLC.FACE-SWAPPER"), None)
                if swapper_mod is not None:
                    swapped_bboxes = []
                    if modules.globals.many_faces and cached_many_faces:
                        result = temp_frame.copy()
                        for t_face in cached_many_faces:
                            result = swapper_mod.swap_face(source_image, t_face, result)
                            if hasattr(t_face, "bbox") and t_face.bbox is not None:
                                swapped_bboxes.append(t_face.bbox.astype(int))
                        temp_frame = result
                    elif cached_target_face is not None:
                        temp_frame = swapper_mod.swap_face(
                            source_image, cached_target_face, temp_frame
                        )
                        if (
                            hasattr(cached_target_face, "bbox")
                            and cached_target_face.bbox is not None
                        ):
                            swapped_bboxes.append(cached_target_face.bbox.astype(int))
                    temp_frame = swapper_mod.apply_post_processing(temp_frame, swapped_bboxes)

                # 2. High-Speed Asynchronous Live Face Enhancer (30-60 FPS smooth, zero lag)
                enhancer_active = (
                    getattr(modules.globals, "live_enhancer_enabled", False)
                    or modules.globals.fp_ui.get("face_enhancer", False)
                    or modules.globals.fp_ui.get("face_enhancer_gpen256", False)
                    or modules.globals.fp_ui.get("face_enhancer_gpen512", False)
                )
                if enhancer_active and cached_faces:
                    from modules.processors.frame.live_enhancer import get_live_enhancer
                    live_enh = get_live_enhancer()
                    cur_model_key = getattr(modules.globals, "live_enhancer_model", "gpen256")
                    faces_to_enhance = cached_many_faces if modules.globals.many_faces else [cached_target_face]
                    for f_obj in faces_to_enhance:
                        if f_obj is not None and hasattr(f_obj, "kps") and f_obj.kps is not None:
                            kps_5 = f_obj.kps.astype(np.float32)
                            if kps_5.shape[0] >= 5:
                                temp_frame = live_enh.apply_to_frame(
                                    temp_frame, kps_5, cur_model_key, modules.globals.source_path
                                )
                        if not modules.globals.many_faces:
                            break
            else:
                modules.globals.target_path = None
                enhancer_active = (
                    getattr(modules.globals, "live_enhancer_enabled", False)
                    or modules.globals.fp_ui.get("face_enhancer", False)
                    or modules.globals.fp_ui.get("face_enhancer_gpen256", False)
                    or modules.globals.fp_ui.get("face_enhancer_gpen512", False)
                )
                if enhancer_active and cached_faces:
                    from modules.processors.frame.live_enhancer import get_live_enhancer
                    live_enh = get_live_enhancer()
                    cur_model_key = getattr(modules.globals, "live_enhancer_model", "gpen256")
                    faces_to_enhance = cached_many_faces if modules.globals.many_faces else [cached_target_face]
                    for f_obj in faces_to_enhance:
                        if f_obj is not None and hasattr(f_obj, "kps") and f_obj.kps is not None:
                            kps_5 = f_obj.kps.astype(np.float32)
                            if kps_5.shape[0] >= 5:
                                temp_frame = live_enh.apply_to_frame(
                                    temp_frame, kps_5, cur_model_key, modules.globals.source_path
                                )
                        if not modules.globals.many_faces:
                            break

            current_time = time.time()
            frame_count += 1
            if current_time - prev_time >= fps_update_interval:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time

            if modules.globals.show_fps:
                cv2.putText(
                    temp_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                )

            # Green line boxes & name badges on both feeds:
            # - Left feed (orig_frame): Green box + "Original" name banner
            # - Right feed (temp_frame): Green box + active swapped face name (e.g. "Baba", "Bhauji") banner
            if modules.globals.show_face_box and cached_faces:
                display_name = get_face_display_name(modules.globals.source_path)
                for face_obj in cached_faces:
                    draw_face_overlay(orig_frame, face_obj, "Original")
                    draw_face_overlay(temp_frame, face_obj, display_name)

            payload = (orig_frame, temp_frame)
            try:
                self._pq.put_nowait(payload)
            except queue.Full:
                try:
                    self._pq.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._pq.put_nowait(payload)
                except queue.Full:
                    pass


class WebcamPreviewWindow(QWidget):
    def __init__(self, camera_index: int):
        super().__init__()
        self.setWindowTitle(f"{modules.metadata.name} — Live Camera Stream")
        self.resize(LIVE_PREVIEW_DEFAULT_WIDTH, LIVE_PREVIEW_DEFAULT_HEIGHT)
        self.setMinimumSize(640, 320)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 8, 10, 10)
        main_layout.setSpacing(6)

        # Control Panel Box (Sleek dark card containing 2 clean, spacious rows)
        ctrl_card = QWidget()
        ctrl_card.setObjectName("controlCard")
        ctrl_card.setStyleSheet(
            "QWidget#controlCard { background-color: #1a1a1a; border: 1px solid #333333; border-radius: 6px; }"
        )
        ctrl_layout = QVBoxLayout(ctrl_card)
        ctrl_layout.setContentsMargins(8, 6, 8, 6)
        ctrl_layout.setSpacing(6)

        # ── Row 1: Face Selector & Auto-Swap Controls ──
        row_face = QHBoxLayout()
        row_face.setContentsMargins(0, 0, 0, 0)
        row_face.setSpacing(8)

        lbl_face = QLabel("🎭 Face:")
        lbl_face.setStyleSheet("font-size: 9.5pt; font-weight: bold; color: #4ade80;")
        row_face.addWidget(lbl_face)

        self._available_faces = get_available_faces()
        self._face_combo = QComboBox()
        self._face_combo.setMinimumWidth(160)
        self._face_combo.setFixedHeight(30)
        self._face_combo.setStyleSheet(
            "QComboBox { background-color: #242424; color: #ffffff; border: 1px solid #404040; "
            "border-radius: 4px; padding: 2px 10px; font-size: 9.5pt; font-weight: 500; }"
            "QComboBox:hover { border-color: #1f538d; }"
            "QComboBox QAbstractItemView { background-color: #242424; color: #ffffff; selection-background-color: #1f538d; }"
        )
        self._current_face_idx = 0
        cur_norm = os.path.normpath(modules.globals.source_path) if modules.globals.source_path else ""
        for i, (dname, fpath) in enumerate(self._available_faces):
            self._face_combo.addItem(dname, fpath)
            if cur_norm and os.path.normpath(fpath) == cur_norm:
                self._current_face_idx = i

        if self._available_faces:
            self._face_combo.setCurrentIndex(self._current_face_idx)
            if not modules.globals.source_path:
                modules.globals.source_path = self._available_faces[self._current_face_idx][1]

        self._face_combo.currentIndexChanged.connect(self._on_face_combo_changed)
        row_face.addWidget(self._face_combo)

        # ◀ Prev Face button
        self._btn_prev = QPushButton("◀ Prev")
        self._btn_prev.setToolTip("Switch to previous face")
        self._btn_prev.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_prev.setFixedHeight(30)
        self._btn_prev.setStyleSheet(
            "QPushButton { background-color: #2b2b2b; color: #ffffff; border: 1px solid #3a3a3a; "
            "border-radius: 4px; padding: 3px 10px; font-size: 9pt; font-weight: 500; }"
            "QPushButton:hover { background-color: #383838; }"
        )
        self._btn_prev.clicked.connect(self._prev_face)
        row_face.addWidget(self._btn_prev)

        # Next Face ▶ button
        self._btn_next = QPushButton("Next ▶")
        self._btn_next.setToolTip("Switch to next face")
        self._btn_next.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_next.setFixedHeight(30)
        self._btn_next.setStyleSheet(
            "QPushButton { background-color: #1f538d; color: #ffffff; border: none; "
            "border-radius: 4px; padding: 3px 12px; font-size: 9pt; font-weight: bold; }"
            "QPushButton:hover { background-color: #2669af; }"
        )
        self._btn_next.clicked.connect(self._next_face)
        row_face.addWidget(self._btn_next)

        # 🔁 Auto-Swap Button
        self._auto_swap_enabled = False
        self._auto_swap_timer = QTimer(self)
        self._auto_swap_timer.timeout.connect(self._next_face)

        self._btn_auto_swap = QPushButton("🔁 Auto-Swap: OFF")
        self._btn_auto_swap.setToolTip("Automatically cycle through faces every 5 seconds")
        self._btn_auto_swap.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_auto_swap.setFixedHeight(30)
        self._btn_auto_swap.setStyleSheet(
            "QPushButton { background-color: #2b2b2b; color: #888888; border: 1px solid #3a3a3a; "
            "border-radius: 4px; padding: 3px 10px; font-size: 9pt; }"
            "QPushButton:hover { background-color: #383838; color: #cccccc; }"
        )
        self._btn_auto_swap.clicked.connect(self._toggle_auto_swap)
        row_face.addWidget(self._btn_auto_swap)

        # 📁 Browse Face Button
        self._btn_browse = QPushButton("📁 Browse...")
        self._btn_browse.setToolTip("Open custom face image file from disk")
        self._btn_browse.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_browse.setFixedHeight(30)
        self._btn_browse.setStyleSheet(
            "QPushButton { background-color: #2b2b2b; color: #cccccc; border: 1px solid #3a3a3a; "
            "border-radius: 4px; padding: 3px 10px; font-size: 9pt; }"
            "QPushButton:hover { background-color: #383838; color: #ffffff; }"
        )
        self._btn_browse.clicked.connect(self._browse_custom_face)
        row_face.addWidget(self._btn_browse)

        row_face.addStretch()

        # Active Target Name Badge
        source_name = get_face_display_name(modules.globals.source_path)
        self._last_source_name = source_name
        self._source_badge = QLabel(f"Active: {source_name}")
        self._source_badge.setStyleSheet(
            "background-color: #1e293b; color: #38bdf8; font-size: 9.5pt; font-weight: bold; "
            "padding: 4px 12px; border-radius: 4px; border: 1px solid #0284c7;"
        )
        row_face.addWidget(self._source_badge)

        ctrl_layout.addLayout(row_face)

        # ── Row 2: Stream View, Bounding Box, Enhancer & Mirror ──
        row_view = QHBoxLayout()
        row_view.setContentsMargins(0, 0, 0, 0)
        row_view.setSpacing(8)

        lbl_view = QLabel("📺 View:")
        lbl_view.setStyleSheet("font-size: 9pt; color: #888888;")
        row_view.addWidget(lbl_view)

        self._active_btn_style = (
            "QPushButton { background-color: #1f538d; color: #ffffff; font-weight: bold; "
            "border-radius: 4px; padding: 4px 10px; font-size: 9pt; border: none; min-height: 24px; }"
            "QPushButton:hover { background-color: #2669af; }"
        )
        self._normal_btn_style = (
            "QPushButton { background-color: #2b2b2b; color: #cccccc; "
            "border-radius: 4px; padding: 4px 10px; font-size: 9pt; border: 1px solid #3a3a3a; min-height: 24px; }"
            "QPushButton:hover { background-color: #383838; color: #ffffff; }"
        )

        self._btn_sbs = QPushButton("◫ Side-by-Side")
        self._btn_sbs.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_sbs.clicked.connect(lambda: self._set_view_mode("side_by_side"))
        row_view.addWidget(self._btn_sbs)

        self._btn_swap = QPushButton("✨ Swapped Only")
        self._btn_swap.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_swap.clicked.connect(lambda: self._set_view_mode("swapped_only"))
        row_view.addWidget(self._btn_swap)

        self._btn_orig = QPushButton("📷 Original Only")
        self._btn_orig.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_orig.clicked.connect(lambda: self._set_view_mode("orig_only"))
        row_view.addWidget(self._btn_orig)

        row_view.addSpacing(8)

        # 🟩 Face Box Toggle
        self._btn_box = QPushButton("🟩 Face Box: ON")
        self._btn_box.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_box.clicked.connect(self._toggle_face_box)
        row_view.addWidget(self._btn_box)
        self._update_face_box_button()

        # ⚡ Enhancer Toggle
        self._btn_enhancer = QPushButton()
        self._btn_enhancer.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_enhancer.clicked.connect(self._cycle_enhancer)
        row_view.addWidget(self._btn_enhancer)
        self._update_enhancer_button()

        # 🪞 Mirror Toggle
        self._btn_mirror = QPushButton("🪞 Mirror: ON" if modules.globals.live_mirror else "🪞 Mirror: OFF")
        self._btn_mirror.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_mirror.clicked.connect(self._toggle_mirror)
        self._btn_mirror.setStyleSheet(
            "QPushButton { background-color: #1f538d; color: #ffffff; font-weight: bold; "
            "border-radius: 4px; padding: 4px 10px; font-size: 9pt; border: none; min-height: 24px; }"
            if modules.globals.live_mirror else
            "QPushButton { background-color: #2b2b2b; color: #cccccc; "
            "border-radius: 4px; padding: 4px 10px; font-size: 9pt; border: 1px solid #3a3a3a; min-height: 24px; }"
        )
        row_view.addWidget(self._btn_mirror)

        row_view.addStretch()

        # 🔄 Reload Faces button
        self._btn_reload = QPushButton("🔄 Reload Faces")
        self._btn_reload.setToolTip("Rescan face folders for new images")
        self._btn_reload.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_reload.setStyleSheet(
            "QPushButton { background-color: #242424; color: #999999; border: 1px solid #333333; "
            "border-radius: 4px; padding: 3px 8px; font-size: 8.5pt; min-height: 24px; }"
            "QPushButton:hover { background-color: #303030; color: #ffffff; }"
        )
        self._btn_reload.clicked.connect(self._reload_faces)
        row_view.addWidget(self._btn_reload)

        ctrl_layout.addLayout(row_view)
        main_layout.addWidget(ctrl_card)

        # Video Layout
        self._video_layout = QHBoxLayout()
        self._video_layout.setContentsMargins(0, 0, 0, 0)
        self._video_layout.setSpacing(10)

        # Left Panel (Original Live)
        self._orig_panel = QWidget()
        orig_vbox = QVBoxLayout(self._orig_panel)
        orig_vbox.setContentsMargins(0, 0, 0, 0)
        orig_vbox.setSpacing(4)
        self._orig_header = QLabel("📷 Original Live Camera")
        self._orig_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._orig_header.setStyleSheet(
            "background-color: #222222; color: #a0a0a0; font-weight: bold; "
            "font-size: 9pt; padding: 4px; border-radius: 3px;"
        )
        orig_vbox.addWidget(self._orig_header)

        self._orig_label = QLabel()
        self._orig_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._orig_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._orig_label.setMinimumSize(1, 1)
        self._orig_label.setStyleSheet("background-color: #0d0d0d; border: 1px solid #2d2d2d; border-radius: 4px;")
        orig_vbox.addWidget(self._orig_label, 1)
        self._video_layout.addWidget(self._orig_panel, 1)

        # Right Panel (Face-Swapped Live)
        self._swap_panel = QWidget()
        swap_vbox = QVBoxLayout(self._swap_panel)
        swap_vbox.setContentsMargins(0, 0, 0, 0)
        swap_vbox.setSpacing(4)
        self._swap_header = QLabel("✨ Face-Swapped Live")
        self._swap_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._swap_header.setStyleSheet(
            "background-color: #1a2a3a; color: #4fa3e3; font-weight: bold; "
            "font-size: 9pt; padding: 4px; border-radius: 3px;"
        )
        swap_vbox.addWidget(self._swap_header)

        self._swap_label = QLabel()
        self._swap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._swap_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._swap_label.setMinimumSize(1, 1)
        self._swap_label.setStyleSheet("background-color: #0d0d0d; border: 1px solid #2d2d2d; border-radius: 4px;")
        swap_vbox.addWidget(self._swap_label, 1)
        self._video_layout.addWidget(self._swap_panel, 1)

        main_layout.addLayout(self._video_layout, 1)

        self._view_mode = "side_by_side"
        self._update_mode_buttons()

        self._cap = VideoCapturer(camera_index)
        req_w, req_h = modules.globals.capture_resolution
        if not self._cap.start(req_w, req_h, 60):
            update_status("Failed to start camera")
            QTimer.singleShot(0, self.close)
            return

        camera_fps = self._cap.actual_fps
        print(
            f"[webcam] Camera running at {self._cap.actual_width}x"
            f"{self._cap.actual_height}@{camera_fps:.0f}fps"
        )

        self._capture_queue: queue.Queue = queue.Queue(maxsize=2)
        self._processed_queue: queue.Queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()

        self._capture_worker = _CaptureWorker(
            self._cap, self._capture_queue, self._stop_event
        )
        self._processing_worker = _ProcessingWorker(
            self._capture_queue, self._processed_queue, self._stop_event, camera_fps
        )
        self._capture_worker.start()
        self._processing_worker.start()

        # Poll at ~2x camera fps so we never block but also don't burn CPU.
        poll_ms = max(1, min(16, int(500 / max(camera_fps, 1))))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(poll_ms)

    def _set_view_mode(self, mode: str) -> None:
        self._view_mode = mode
        if mode == "side_by_side":
            self._orig_panel.setVisible(True)
            self._swap_panel.setVisible(True)
            self._orig_header.setVisible(True)
            self._swap_header.setVisible(True)
        elif mode == "swapped_only":
            self._orig_panel.setVisible(False)
            self._swap_panel.setVisible(True)
            self._swap_header.setVisible(False)
        elif mode == "orig_only":
            self._orig_panel.setVisible(True)
            self._swap_panel.setVisible(False)
            self._orig_header.setVisible(False)
        self._update_mode_buttons()

    def _update_mode_buttons(self) -> None:
        self._btn_sbs.setStyleSheet(
            self._active_btn_style if self._view_mode == "side_by_side" else self._normal_btn_style
        )
        self._btn_swap.setStyleSheet(
            self._active_btn_style if self._view_mode == "swapped_only" else self._normal_btn_style
        )
        self._btn_orig.setStyleSheet(
            self._active_btn_style if self._view_mode == "orig_only" else self._normal_btn_style
        )

    def _toggle_face_box(self) -> None:
        modules.globals.show_face_box = not modules.globals.show_face_box
        self._update_face_box_button()

    def _update_face_box_button(self) -> None:
        if getattr(modules.globals, "show_face_box", True):
            self._btn_box.setText("🟩 Face Box: ON")
            self._btn_box.setStyleSheet(
                "QPushButton { background-color: #1e3a29; color: #4ade80; border: 1px solid #16a34a; "
                "border-radius: 4px; padding: 4px 10px; font-size: 9pt; font-weight: bold; }"
                "QPushButton:hover { background-color: #166534; }"
            )
        else:
            self._btn_box.setText("⬜ Face Box: OFF")
            self._btn_box.setStyleSheet(
                "QPushButton { background-color: #2b2b2b; color: #888888; border: 1px solid #3a3a3a; "
                "border-radius: 4px; padding: 4px 10px; font-size: 9pt; }"
                "QPushButton:hover { background-color: #383838; color: #cccccc; }"
            )

    def _cycle_enhancer(self) -> None:
        # Cycle: OFF -> Fast (GPEN-256) -> Balanced (GPEN-512) -> Quality (GFPGAN) -> OFF
        is_on = getattr(modules.globals, "live_enhancer_enabled", False)
        cur_model = getattr(modules.globals, "live_enhancer_model", "gpen256")
        if not is_on:
            modules.globals.live_enhancer_enabled = True
            modules.globals.live_enhancer_model = "gpen256"
        elif cur_model == "gpen256":
            modules.globals.live_enhancer_model = "gpen512"
        elif cur_model == "gpen512":
            modules.globals.live_enhancer_model = "gfpgan"
        else:
            modules.globals.live_enhancer_enabled = False
            modules.globals.live_enhancer_model = "gpen256"
        modules.globals.fp_ui["face_enhancer"] = modules.globals.live_enhancer_enabled
        self._update_enhancer_button()

    def _update_enhancer_button(self) -> None:
        is_on = getattr(modules.globals, "live_enhancer_enabled", False)
        cur_model = getattr(modules.globals, "live_enhancer_model", "gpen256")
        if not is_on:
            self._btn_enhancer.setText("⚪ Enhancer: OFF")
            self._btn_enhancer.setStyleSheet(
                "QPushButton { background-color: #2b2b2b; color: #888888; border: 1px solid #3a3a3a; "
                "border-radius: 4px; padding: 4px 10px; font-size: 9pt; }"
                "QPushButton:hover { background-color: #383838; color: #cccccc; }"
            )
        elif cur_model == "gpen256":
            self._btn_enhancer.setText("⚡ Enhancer: Fast")
            self._btn_enhancer.setStyleSheet(
                "QPushButton { background-color: #1e3a29; color: #4ade80; border: 1px solid #16a34a; "
                "border-radius: 4px; padding: 4px 10px; font-size: 9pt; font-weight: bold; }"
                "QPushButton:hover { background-color: #166534; }"
            )
        elif cur_model == "gpen512":
            self._btn_enhancer.setText("🎯 Enhancer: Balanced")
            self._btn_enhancer.setStyleSheet(
                "QPushButton { background-color: #1a2a3a; color: #38bdf8; border: 1px solid #0284c7; "
                "border-radius: 4px; padding: 4px 10px; font-size: 9pt; font-weight: bold; }"
                "QPushButton:hover { background-color: #0369a1; }"
            )
        else:
            self._btn_enhancer.setText("💎 Enhancer: Quality")
            self._btn_enhancer.setStyleSheet(
                "QPushButton { background-color: #2e1065; color: #c084fc; border: 1px solid #7c3aed; "
                "border-radius: 4px; padding: 4px 10px; font-size: 9pt; font-weight: bold; }"
                "QPushButton:hover { background-color: #3b0764; }"
            )

    def _toggle_auto_swap(self) -> None:
        self._auto_swap_enabled = not getattr(self, "_auto_swap_enabled", False)
        if self._auto_swap_enabled:
            self._auto_swap_timer.start(5000)
            self._btn_auto_swap.setText("🔁 Auto-Swap: ON (5s)")
            self._btn_auto_swap.setStyleSheet(
                "QPushButton { background-color: #7c2d12; color: #fdba74; border: 1px solid #ea580c; "
                "border-radius: 4px; padding: 3px 8px; font-size: 9pt; font-weight: bold; }"
                "QPushButton:hover { background-color: #9a3412; }"
            )
        else:
            self._auto_swap_timer.stop()
            self._btn_auto_swap.setText("🔁 Auto-Swap: OFF")
            self._btn_auto_swap.setStyleSheet(
                "QPushButton { background-color: #2b2b2b; color: #888888; border: 1px solid #3a3a3a; "
                "border-radius: 4px; padding: 3px 8px; font-size: 9pt; }"
                "QPushButton:hover { background-color: #383838; color: #cccccc; }"
            )

    def _prev_face(self) -> None:
        if not self._available_faces:
            return
        self._current_face_idx = (self._current_face_idx - 1) % len(self._available_faces)
        self._apply_face_index(self._current_face_idx)

    def _next_face(self) -> None:
        if not self._available_faces:
            return
        self._current_face_idx = (self._current_face_idx + 1) % len(self._available_faces)
        self._apply_face_index(self._current_face_idx)

    def _apply_face_index(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._available_faces):
            return
        self._current_face_idx = idx
        dname, fpath = self._available_faces[idx]
        norm_path = os.path.normpath(fpath)
        modules.globals.source_path = norm_path
        self._source_badge.setText(f"Active: {dname}")
        self._face_combo.blockSignals(True)
        self._face_combo.setCurrentIndex(idx)
        self._face_combo.blockSignals(False)
        # Sync with MainWindow if open
        if _MAIN is not None and hasattr(_MAIN, "sync_source_face"):
            _MAIN.sync_source_face(norm_path)

    def _on_face_combo_changed(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._available_faces):
            return
        self._apply_face_index(idx)

    def _browse_custom_face(self) -> None:
        global _RECENT_SOURCE_DIR
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Custom Face Image",
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if path and is_image(path):
            norm_path = os.path.normpath(path)
            _RECENT_SOURCE_DIR = os.path.dirname(norm_path)
            modules.globals.source_path = norm_path
            for i, (_, fpath) in enumerate(self._available_faces):
                if os.path.normpath(fpath) == norm_path:
                    self._apply_face_index(i)
                    return
            dname = get_face_display_name(norm_path)
            self._available_faces.append((dname, norm_path))
            self._face_combo.addItem(dname, norm_path)
            self._apply_face_index(len(self._available_faces) - 1)

    def _reload_faces(self) -> None:
        self._available_faces = get_available_faces()
        cur_norm = os.path.normpath(modules.globals.source_path) if modules.globals.source_path else ""
        self._face_combo.blockSignals(True)
        self._face_combo.clear()
        selected_idx = 0
        for i, (dname, fpath) in enumerate(self._available_faces):
            self._face_combo.addItem(dname, fpath)
            if cur_norm and os.path.normpath(fpath) == cur_norm:
                selected_idx = i
        if self._available_faces:
            self._face_combo.setCurrentIndex(selected_idx)
            modules.globals.source_path = self._available_faces[selected_idx][1]
            dname = self._available_faces[selected_idx][0]
            self._source_badge.setText(f"Active: {dname}")
        self._face_combo.blockSignals(False)

    def _toggle_mirror(self) -> None:
        modules.globals.live_mirror = not modules.globals.live_mirror
        self._btn_mirror.setText("🪞 Mirror: ON" if modules.globals.live_mirror else "🪞 Mirror: OFF")
        self._btn_mirror.setStyleSheet(
            "QPushButton { background-color: #1f538d; color: #ffffff; font-weight: bold; "
            "border-radius: 4px; padding: 4px 10px; font-size: 9pt; border: none; min-height: 24px; }"
            if modules.globals.live_mirror else
            "QPushButton { background-color: #2b2b2b; color: #cccccc; "
            "border-radius: 4px; padding: 4px 10px; font-size: 9pt; border: 1px solid #3a3a3a; min-height: 24px; }"
        )

    def _apply_source_path_external(self, norm_path: str) -> None:
        """Called when MainWindow changes the active face."""
        cur_norm = os.path.normpath(norm_path)
        modules.globals.source_path = cur_norm
        found_idx = -1
        for i, (_, fpath) in enumerate(self._available_faces):
            if os.path.normpath(fpath) == cur_norm:
                found_idx = i
                break
        if found_idx == -1:
            dname = get_face_display_name(cur_norm)
            self._available_faces.append((dname, cur_norm))
            self._face_combo.addItem(dname, cur_norm)
            found_idx = self._face_combo.count() - 1

        self._current_face_idx = found_idx
        self._face_combo.blockSignals(True)
        self._face_combo.setCurrentIndex(found_idx)
        self._face_combo.blockSignals(False)
        dname = self._available_faces[found_idx][0]
        self._source_badge.setText(f"Active: {dname}")

    def _tick(self) -> None:
        if self._stop_event.is_set():
            self.close()
            return
        try:
            payload = self._processed_queue.get_nowait()
        except queue.Empty:
            return

        if isinstance(payload, tuple):
            orig_frame, swap_frame = payload
        else:
            orig_frame = payload
            swap_frame = payload

        # Keep target face badge and combo updated if source_path changed
        if modules.globals.source_path:
            cur_name = get_face_display_name(modules.globals.source_path)
            if self._last_source_name != cur_name:
                self._last_source_name = cur_name
                self._source_badge.setText(f"Active: {cur_name}")
                norm_src = os.path.normpath(modules.globals.source_path)
                for i, (_, fpath) in enumerate(self._available_faces):
                    if os.path.normpath(fpath) == norm_src:
                        self._face_combo.blockSignals(True)
                        self._face_combo.setCurrentIndex(i)
                        self._current_face_idx = i
                        self._face_combo.blockSignals(False)
                        break

        if self._view_mode == "side_by_side":
            ow = max(10, self._orig_label.width())
            oh = max(10, self._orig_label.height())
            sw = max(10, self._swap_label.width())
            sh = max(10, self._swap_label.height())

            orig_scaled = fit_image_to_size(orig_frame, ow, oh)
            swap_scaled = fit_image_to_size(swap_frame, sw, sh)

            self._orig_label.setPixmap(_bgr_to_qpixmap(orig_scaled))
            self._swap_label.setPixmap(_bgr_to_qpixmap(swap_scaled))

        elif self._view_mode == "swapped_only":
            sw = max(10, self._swap_label.width())
            sh = max(10, self._swap_label.height())
            swap_scaled = fit_image_to_size(swap_frame, sw, sh)
            self._swap_label.setPixmap(_bgr_to_qpixmap(swap_scaled))

        elif self._view_mode == "orig_only":
            ow = max(10, self._orig_label.width())
            oh = max(10, self._orig_label.height())
            orig_scaled = fit_image_to_size(orig_frame, ow, oh)
            self._orig_label.setPixmap(_bgr_to_qpixmap(orig_scaled))

    def closeEvent(self, event) -> None:
        self._stop_event.set()
        try:
            self._timer.stop()
        except Exception:
            pass
        for worker in (self._capture_worker, self._processing_worker):
            try:
                worker.wait(2000)
            except Exception:
                pass
        try:
            self._cap.release()
        except Exception:
            pass
        global _WEBCAM_PREVIEW
        if _WEBCAM_PREVIEW is self:
            _WEBCAM_PREVIEW = None
        event.accept()


def _open_webcam_preview(camera_index: int) -> None:
    global _WEBCAM_PREVIEW
    if _WEBCAM_PREVIEW is not None:
        _WEBCAM_PREVIEW.close()
    _WEBCAM_PREVIEW = WebcamPreviewWindow(camera_index)
    _WEBCAM_PREVIEW.show()


# ─── mapper dialogs (image/video + live) ────────────────────────────────


def _make_thumb(cv2_img: np.ndarray) -> QPixmap:
    rgb = gpu_cvt_color(cv2_img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).resize(
        (MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE), Image.LANCZOS
    )
    return _pil_to_qpixmap(image)


class MapperDialog(QDialog):
    """Source × Target mapper for image / video processing."""

    def __init__(self, start_cb: Callable, mapping: list):
        super().__init__(_MAIN)
        self._start_cb = start_cb
        self._map = mapping
        self.setWindowTitle(_("Source x Target Mapper"))
        self.resize(POPUP_WIDTH, POPUP_HEIGHT)
        layout = QVBoxLayout(self)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_submit = QPushButton(_("Submit"))
        btn_submit.clicked.connect(self._on_submit)
        layout.addWidget(btn_submit, alignment=Qt.AlignmentFlag.AlignCenter)

        self._rebuild()

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for item in self._map:
            row = item["id"]
            btn = QPushButton(_("Select source image"))
            btn.setFixedWidth(200)
            btn.clicked.connect(lambda _c, n=row: self._select_source(n))
            grid.addWidget(btn, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(_make_thumb(item["source"]["cv2"]))
                src_label.setText("")

            x_label = QLabel("×")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setStyleSheet("border: 1px solid #555;")
            grid.addWidget(tgt_label, row, 3)
            if "target" in item:
                tgt_label.setPixmap(_make_thumb(item["target"]["cv2"]))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_source(self, row: int) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if not path:
            return
        cv2_img = imread_unicode(path)
        face = get_one_face(cv2_img)
        if face is None:
            self.set_status("Face could not be detected in last upload!")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row]["source"] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_submit(self) -> None:
        if has_valid_map():
            self.accept()
            _MAIN._select_output_and_start()
        else:
            self.set_status("Atleast 1 source with target is required!")


class LiveMapperDialog(QDialog):
    """Source × Target mapper for live webcam mode."""

    def __init__(self, camera_index: int, mapping: list):
        super().__init__(_MAIN)
        self._camera_index = camera_index
        self._map = mapping
        self.setWindowTitle(_("Source x Target Mapper"))
        self.resize(POPUP_LIVE_WIDTH, POPUP_LIVE_HEIGHT)
        layout = QVBoxLayout(self)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        layout.addWidget(self._scroll, 1)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        btn_row = QHBoxLayout()
        for text, slot in (
            (_("Add"), self._on_add),
            (_("Clear"), self._on_clear),
            (_("Submit"), self._on_submit),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        self._rebuild()

    def set_status(self, text: str) -> None:
        self._status.setText(_(text))

    def _rebuild(self) -> None:
        body = QWidget()
        grid = QGridLayout(body)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for item in self._map:
            row = item["id"]
            btn_s = QPushButton(_("Select source image"))
            btn_s.setFixedWidth(200)
            btn_s.clicked.connect(lambda _c, n=row: self._select_face(n, "source"))
            grid.addWidget(btn_s, row, 0)

            src_label = QLabel(f"S-{row}")
            src_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            src_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            src_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(src_label, row, 1)
            if "source" in item:
                src_label.setPixmap(_make_thumb(item["source"]["cv2"]))
                src_label.setText("")

            x_label = QLabel("×")
            x_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(x_label, row, 2)

            btn_t = QPushButton(_("Select target image"))
            btn_t.setFixedWidth(200)
            btn_t.clicked.connect(lambda _c, n=row: self._select_face(n, "target"))
            grid.addWidget(btn_t, row, 3)

            tgt_label = QLabel(f"T-{row}")
            tgt_label.setFixedSize(MAPPER_PREVIEW_SIZE, MAPPER_PREVIEW_SIZE)
            tgt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tgt_label.setStyleSheet("border: 1px dashed #555;")
            grid.addWidget(tgt_label, row, 4)
            if "target" in item:
                tgt_label.setPixmap(_make_thumb(item["target"]["cv2"]))
                tgt_label.setText("")

        grid.setRowStretch(grid.rowCount(), 1)
        self._scroll.setWidget(body)

    def _select_face(self, row: int, kind: str) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, _("select an source image"),
            _RECENT_SOURCE_DIR or "",
            _IMAGE_FILE_FILTER,
        )
        if not path:
            return
        cv2_img = imread_unicode(path)
        face = get_one_face(cv2_img)
        if face is None:
            self.set_status("Face could not be detected in last upload!")
            return
        x_min, y_min, x_max, y_max = face["bbox"]
        self._map[row][kind] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        self._rebuild()

    def _on_add(self) -> None:
        add_blank_map()
        self._rebuild()
        self.set_status("Please provide mapping!")

    def _on_clear(self) -> None:
        for item in self._map:
            item.pop("source", None)
            item.pop("target", None)
        self._rebuild()
        self.set_status("All mappings cleared!")

    def _on_submit(self) -> None:
        if has_valid_map():
            simplify_maps()
            self.set_status("Mappings successfully submitted!")
            self.accept()
            _open_webcam_preview(self._camera_index)
        else:
            self.set_status("At least 1 source with target is required!")


def _open_mapper_dialog(start_cb: Callable, mapping: list) -> None:
    global _MAPPER
    close_mapper_window()
    _MAPPER = MapperDialog(start_cb, mapping)
    _MAPPER.show()


def _open_live_mapper_dialog(camera_index: int, mapping: list) -> None:
    global _LIVE_MAPPER
    close_mapper_window()
    _LIVE_MAPPER = LiveMapperDialog(camera_index, mapping)
    _LIVE_MAPPER.show()


def close_mapper_window() -> None:
    global _MAPPER, _LIVE_MAPPER
    if _MAPPER is not None:
        _MAPPER.close()
        _MAPPER = None
    if _LIVE_MAPPER is not None:
        _LIVE_MAPPER.close()
        _LIVE_MAPPER = None


# ─── entry point ─────────────────────────────────────────────────────────


class _Window:
    """Thin wrapper exposing .mainloop() for core.py compatibility."""

    def __init__(self, app: QApplication, main_window: MainWindow):
        self._app = app
        self._main = main_window

    def mainloop(self) -> None:
        self._main.show()
        self._app.exec()


def init(
    start: Callable[[], None], destroy: Callable[[], None], lang: str
) -> _Window:
    global _APP, _MAIN, _PREVIEW, _LANG, _BRIDGE

    _LANG = LanguageManager(lang)
    if QApplication.instance() is None:
        _APP = QApplication(sys.argv)
    else:
        _APP = QApplication.instance()
    _APP.setStyleSheet(QSS)

    _BRIDGE = _UIBridge()
    _MAIN = MainWindow(start, destroy)
    _PREVIEW = PreviewWindow()

    # Route status updates onto the UI thread regardless of caller.
    _BRIDGE.statusChanged.connect(_MAIN.set_status)

    return _Window(_APP, _MAIN)
