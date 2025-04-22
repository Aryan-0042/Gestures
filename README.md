# AI-Based Hand Gesture Control System

An **AI-driven** project that lets users control their computer through **hand gestures** instead of a traditional mouse. It leverages **MediaPipe** for real-time hand tracking and a **Multi-Layer Perceptron (MLP)** for gesture classification, providing multiple modes (cursor control, presentation control, teaching/whiteboard) with smooth cursor movement, clicks, and scrolling.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Collection & Training](#data-collection--training)
- [Usage](#usage)

---

## Overview
This project aims to replace traditional mouse interactions with **gesture-based** controls. Using **MediaPipe** for hand landmarks, an MLP classifies gestures like **cursor_move**, **left_click**, **right_click**, and **scroll**. The system can be extended to **presentation mode** (slide control) and **teaching mode** (virtual whiteboard).

### Why a Local Dataset?
- **Custom Gestures:** Define exactly which gestures you need.  
- **Better Accuracy:** Data collected under your real-world conditions.  
- **Full Control:** Ensure the model labels match your code logic.

---

## Features
1. **Cursor Control Mode**  
   - **Smooth Movement:** Exponential smoothing for stable cursor actions.  
   - **Click & Drag:** One-time click actions, hold-click logic for dragging.  
   - **Scroll:** Speed increases the longer you hold a scroll gesture.

2. **Presentation Mode** (Blueprint)  
   - Slide navigation (next/previous), zoom.  
   - Real-time gesture overlay.

3. **Teaching Mode** (Blueprint)  
   - Whiteboard drawing, erase, clear-canvas gestures.  
   - Potential for collaborative or advanced drawing logic.

4. **Real-Time Feedback**  
   - Displays the recognized gesture text in the OpenCV window.  
   - Visualizes hand landmarks for user clarity.

---

## Project Structure

1.  **gesture\_classifier.py**
    -   Loads the trained **TFLite model** (`gesture_model.tflite`) and its **label encoder** (`gesture_label_encoder.pkl`).
    -   Exposes a `predict_gesture(landmarks: List[float]) -> str` method that returns the gesture name given a 42-value landmark vector.

2.  **cursor\_control.py**
    -   Defines the `CursorControl` class, which:
        -   Uses **MediaPipe** to detect hand landmarks in real time.
        -   Calls `GestureClassifier` to predict gestures.
        -   Maps gestures to OS mouse actions (**move**, **left/right/double click**, **scroll**).
        -   Smooths cursor motion and implements **velocity-based scrolling**.

3.  **presentation\_control.py**
    -   Defines the `PresentationControl` class, which:
        -   Detects gestures via **MediaPipe** + `GestureClassifier`.
        -   Debounces slide navigation (**next/previous**) with a **4-second timeout**.
        -   Debounces **zoom in/out** with a **2-second timeout**.
        -   Supports **starting (F5)** and **exiting (Esc) slideshow** with customizable cooldowns.

4.  **teaching\_control.py**
    -   Defines the `TeachingControl` class, which:
        -   Tracks hand landmarks and predicts gestures.
        -   Manages a persistent **whiteboard canvas** (**draw**, **erase**).
        -   Supports **undo/redo stacks** and a cooldown-protected "**clear canvas**" gesture.

5.  **train\_model.py**
    -   Loads raw landmark **CSV data** (e.g., `gestures.csv`).
    -   Builds, trains, and validates an **MLP classifier** (**TensorFlow/Keras**).
    -   Saves outputs:
        -   `gesture_model.tflite` (for fast inference)
        -   `gesture_label_encoder.pkl` (to map model outputs back to gesture names)

6.  **data\_collection.py**
    -   Captures webcam frames via **OpenCV** and **MediaPipe**.
    -   Prompts the user for a gesture label, records the **21 hand-landmark pairs**, and appends them to `gestures.csv`.
    -   Facilitates **balanced, labeled dataset creation**.

7.  **main.py**
    -   Implements the **Tkinter GUI**:
        -   **Home Page** with navigation buttons and "**About**" info.
        -   **Cursor**, **Presentation**, **Teaching pages**—each embedding a live **OpenCV feed**, **Start/Stop toggle**, **Overlay toggle**, and an in-place "**Info**" overlay grid.
        -   Dynamically loads and invokes `CursorControl`, `PresentationControl`, and `TeachingControl` logic via a shared `ModePageBase`.

8.  **requirements.txt**
    ```text
    opencv-python
    mediapipe
    tensorflow
    pyautogui
    pillow
    numpy
    ```
    -   Pinpoints all **Python dependencies**.
    -   Ensures reproducible environment setup using `pip install -r requirements.txt`.
---

## Installation

1.  **Clone repository**
    ```bash
    git clone [https://github.com/Aryan-0042/Gestures.git](https://github.com/Aryan-0042/Gestures.git)
    cd Gestures
    ```

2.  **Create virtual environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # Linux/MacOS
    venv\Scripts\activate      # Windows
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    
---

## Data Collection & Training

1.  **Collect gestures**
    python data_collection.py --gesture cursor_move --output gestures.csv

2.  **Balance & clean** (`gestures.csv`) for uniform class samples.

3.  **Train MLP & export**
    python train_model.py --input gestures.csv --output gesture_model.h5


4.  **Convert to TFLite** (inside train script or via converter API).

---


## Usage
Run:
```bash
python main.py
```
**Home Screen**: Select one of three modes.
**Start/Stop**: Begin or end webcam-based detection.
**Toggle Overlay**: Show/hide landmarks and gesture labels.
**Info**: Displays mode-specific supported gestures and images.
**Back**: Return to Home.\n'


---

## Contributing

1.  **Fork the repository**
2.  **Create a branch**:
    git checkout -b feature/MyFeature
	
3.  **Commit changes**:
    git commit -m "Add MyFeature"
4.  **Push to your fork**:
    git push origin feature/MyFeature
    
5.  **Open a Pull Request**