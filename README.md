# 🎭 LiveFace-Studio (v2.2.0)

<p align="center">
  <strong>Next-Generation Real-Time AI Face Swapping & Live Camera Studio</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?style=for-the-badge&logo=python" alt="Python Version"/>
  <img src="https://img.shields.io/badge/CUDA-12.x%20Accelerated-green?style=for-the-badge&logo=nvidia" alt="CUDA"/>
  <img src="https://img.shields.io/badge/UI-PySide6%20Qt-blueviolet?style=for-the-badge" alt="PySide6"/>
  <img src="https://img.shields.io/badge/Platform-Windows%2010%20%2F%2011-0078D6?style=for-the-badge&logo=windows" alt="Windows"/>
</p>

---

## 🌟 Overview

**LiveFace-Studio** is a high-performance, real-time AI face swapping application built specifically for live camera streams and video production. Designed with an ultra-responsive PySide6 interface and GPU-accelerated inference pipelines, LiveFace-Studio lets you swap and enhance faces live with zero latency.

---

## ✨ Key Features

### 📺 1. Dual Live Camera Stream (Side-by-Side)
- **Simultaneous Monitoring**: Compare the **Original Live Camera** feed on the left and the **Face-Swapped Live Camera** feed on the right in real time.
- **Flexible Layouts**:
  - `[ ◫ Side-by-Side ]` — Dual monitoring mode.
  - `[ ✨ Swapped Only ]` — Fullscreen view of the face-swapped stream.
  - `[ 📷 Original Only ]` — View the clean live camera stream.

### 🎯 2. High-Tech Green HUD Bounding Boxes
- **Intelligent Overlays**: Dynamic green bounding boxes with corner HUD brackets track faces accurately on both feeds.
- **Clear Identification**:
  - Left Feed: Displays the **`Original`** tag with high-contrast badge.
  - Right Feed: Displays the active swap face name (e.g. **`Adi`**, **`Baba`**, **`Bhauji`**).
- **One-Click Toggle**: Enable or disable bounding boxes instantly using `[ 🟩 Face Box: ON / OFF ]`.

### 🔁 3. Preset Faces & Automated Cycling
- **Built-in Face Selector**: Quickly switch between target faces using the toolbar dropdown.
- **Navigation Controls**: `[ ◀ Prev ]` and `[ Next ▶ ]` buttons allow instant browsing.
- **Folder Browser**: Click `[ 📁 Browse... ]` to load any custom folder of portrait images.
- **Auto-Swap**: Toggle `[ 🔁 Auto-Swap: ON / OFF ]` to automatically cycle between faces during a live broadcast.

### ⚡ 4. Asynchronous Live Enhancer (GPEN-BFR-256)
- **Zero-Stutter Performance**: Deep GAN enhancement runs in a dedicated background worker thread on CUDA.
- **Smooth 30–60 FPS**: Live video frames are continuously rendered without camera lag or head-motion delay.
- **Instant Toggle**: Switch between `[ Enhancer: OFF ]` and `[ Enhancer: GPEN-256 ]` on the fly.

### 🪞 5. Real-Time Camera & Display Controls
- **Live Mirror**: Flip the webcam feed horizontally for natural viewing.
- **Camera Selection**: Supports integrated webcams, USB cameras, and virtual cameras (e.g., OBS Virtual Camera, Iriun Webcam).
- **High Resolution Support**: Dynamically scales and fits any camera resolution with crisp aspect ratio preservation.

---

## 🚀 Quick Start Guide

### 1. One-Click Automated Setup
Before launching the application for the first time, run the automated setup script:

```bat
setup.bat
```

**What `setup.bat` does automatically:**
1. Checks your Python environment (Python 3.10, 3.11, or 3.12).
2. Initializes the virtual environment in `.\venv`.
3. Upgrades core package tools (`pip`, `wheel`, `setuptools`).
4. Installs all required packages from `requirements.txt`.
5. Pre-downloads all necessary AI models:
   - `inswapper_128_fp16.onnx` (High-speed FP16 face swapper)
   - `buffalo_l` InsightFace model pack
   - `GPEN-BFR-256.onnx` (Real-time face restoration model)
6. Verifies CUDA / GPU acceleration and camera availability.

---

### 2. Launching LiveFace-Studio
Once setup is complete, launch the studio with:

```bat
run.bat
```

No complicated command-line parameters needed! `run.bat` automatically:
- Activates UTF-8 encoding.
- Configures CUDA / GPU hardware acceleration.
- Launches the studio.

---

## 🎛️ Toolbar Controls Guide

| Control | Description |
| :--- | :--- |
| **`Face:` [Dropdown]** | Select the active target face from your loaded preset library. |
| **`[ ◀ Prev ]` / `[ Next ▶ ]`** | Step to the previous or next face in the collection. |
| **`[ 📁 Browse... ]`** | Select a custom folder containing face images (`.jpg`, `.png`). |
| **`[ 🔁 Auto-Swap ]`** | Automatically cycle through faces at timed intervals. |
| **`[ ◫ Side-by-Side ]`** | Show both Original and Swapped feeds side by side. |
| **`[ ✨ Swapped Only ]`** | Show only the face-swapped output feed. |
| **`[ 📷 Original Only ]`** | Show only the live webcam feed. |
| **`[ 🟩 Face Box ]`** | Toggle green HUD bounding boxes and face name badges. |
| **`[ ⚡ Enhancer ]`** | Toggle GPEN-BFR-256 real-time face resolution enhancement. |
| **`[ 🪞 Mirror ]`** | Flip camera output horizontally. |

---

## 📂 Project Architecture

```
LiveFace-Studio/
├── run.bat                 # Unified one-click application launcher
├── setup.bat               # Automated environment & model installer
├── setup_env.py            # Environment validation & model downloader
├── requirements.txt        # Python dependency manifest
├── pyproject.toml          # Project configuration
├── ffmpeg.exe              # Bundled high-performance media encoder
├── ffprobe.exe             # Bundled media analysis tool
├── models/                 # AI model repository
│   ├── inswapper_128_fp16.onnx
│   ├── GPEN-BFR-256.onnx
│   └── GPEN-BFR-512.onnx
├── modules/                # Core studio modules
│   ├── ui.py               # PySide6 Dual-Screen Live Preview UI
│   ├── core.py             # Application lifecycle & coordinator
│   ├── face_analyser.py    # InsightFace detection & landmarks
│   ├── model_downloader.py # Automated model fetcher
│   ├── platform_info.py    # Hardware & accelerator auto-detection
│   └── processors/frame/
│       ├── face_swapper.py # Core swapping & Poisson blending
│       └── live_enhancer.py# Asynchronous background GPEN enhancer
└── venv/                   # Python virtual environment
```

---

## 💻 Hardware Requirements

- **Operating System**: Windows 10 or Windows 11 (64-bit).
- **Processor**: Intel Core i5 / AMD Ryzen 5 or better.
- **GPU**: NVIDIA GeForce GTX 1060 or newer (RTX 2060/3060+ recommended for real-time 60 FPS).
- **RAM**: 8 GB minimum (16 GB recommended).
- **Python**: 3.10 – 3.12 (64-bit).

---

## ⚖️ Ethics & Disclaimer

LiveFace-Studio is developed for creative media, virtual streaming, digital avatars, and computer vision research:
- **Consent**: Obtain consent from any individual before using their likeness.
- **Transparency**: Clearly indicate when live video feeds or media contain AI-synthesized faces.
- **Safety**: Do not use this software for deceptive, defamatory, or unlawful activities.
