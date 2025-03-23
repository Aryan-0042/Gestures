# AI-Based Hand Gesture Control System

An **AI-driven** project that lets users control their computer through **hand gestures** instead of a traditional mouse. It leverages **MediaPipe** for real-time hand tracking and a **Multi-Layer Perceptron (MLP)** for gesture classification, providing multiple modes (cursor control, presentation control, teaching/whiteboard) with smooth cursor movement, clicks, and scrolling.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Collection & Training](#data-collection--training)
- [Usage](#usage)
- [Packaging](#packaging)
- [Contributing](#contributing)
- [License](#license)

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

1. **gesture_classifier.py**  
   - Contains the **MLP model** loading (`gesture_model.h5`) and a function like `predict_gesture()` to classify landmark inputs.

2. **cursor_control.py**  
   - Implements **cursor movement** with exponential smoothing, click/drag logic, and optional scrolling.

3. **presentation_control.py** (Blueprint)  
   - Future expansion for **slide navigation**, zoom in/out, or pointer actions during presentations.

4. **teaching_control.py** (Blueprint)  
   - Plans for a **whiteboard** interface: drawing, erasing, clearing the canvas, etc.

5. **train_model.py**  
   - Loads your CSV gesture data, trains a Multi-Layer Perceptron, and saves `gesture_model.h5` + `gesture_label_encoder.pkl`.

6. **data_collection.py**  
   - Captures **hand landmarks** with MediaPipe, appends labeled data to a CSV (e.g., `gestures.csv`).

7. **main.py**  
   - **Tkinter UI** that imports the above modules and runs the system.

8. **requirements.txt**  
   - Optional file listing all needed libraries (OpenCV, MediaPipe, TensorFlow, etc.).

---
