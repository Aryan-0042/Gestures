# teaching_control.py

import cv2
import mediapipe as mp
import numpy as np
import time
from gesture_classifier import GestureClassifier

class TeachingControl:
    def __init__(self, show_gesture=True):
        self.classifier = GestureClassifier(
            tflite_path="gesture_model.tflite",
            encoder_path="gesture_label_encoder.pkl"
        )
        self.mp_hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.canvas = None
        self.strokes = []
        self.redo_stack = []

        self.draw_color = (255, 255, 255)
        self.erase_color = (0, 0, 0)
        self.draw_thickness = 3
        self.erase_thickness = 30

        self.prev_point = None
        self.last_gesture = "no_gesture"

        self.show_gesture = show_gesture
        self.last_clear_time = 0
        self.clear_cooldown = 3 #delay

        # *** New: current‐color display ***
        self.current_color_label = None

    def process_frame(self, frame, overlay=True):
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb)

        gesture = "no_gesture"
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            if overlay:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                )
            pts = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y)]
            gesture = self.classifier.predict_gesture(pts)
            self._perform_action(gesture, frame, hand_landmarks)
        else:
            self._perform_action("no_gesture", frame, None)

        blended = cv2.addWeighted(frame, 0.5, self.canvas, 0.5, 0)

        if overlay and self.show_gesture:
            label_to_show = self.label_mapper(gesture) if hasattr(self, "label_mapper") else gesture
            cv2.putText(
                blended, f"Gesture: {label_to_show}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        return blended

    def _perform_action(self, gesture, frame, hand_landmarks):
        now = time.time()
        current_pt = None

        if hand_landmarks:
            lm = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            current_pt = (int(lm.x * w), int(lm.y * h))

        # DRAW stroke
        if gesture == "cursor_move":
            if self.last_gesture != "cursor_move":
                self.strokes.append({
                    "pts": [], "color": self.draw_color, "thick": self.draw_thickness
                })
                self.redo_stack.clear()
                self.prev_point = None
            if current_pt:
                stroke = self.strokes[-1]
                stroke["pts"].append(current_pt)
                if self.prev_point:
                    cv2.line(
                        self.canvas, self.prev_point, current_pt,
                        stroke["color"], stroke["thick"]
                    )
                self.prev_point = current_pt

        # ERASE stroke
        elif gesture == "double_click":
            if self.last_gesture != "double_click":
                self.strokes.append({
                    "pts": [], "color": self.erase_color, "thick": self.erase_thickness
                })
                self.redo_stack.clear()
                self.prev_point = None
            if current_pt:
                stroke = self.strokes[-1]
                stroke["pts"].append(current_pt)
                if self.prev_point:
                    cv2.line(
                        self.canvas, self.prev_point, current_pt,
                        stroke["color"], stroke["thick"]
                    )
                self.prev_point = current_pt

        # CLEAR canvas with cooldown
        elif gesture == "scroll_up":
            if (now - self.last_clear_time) >= self.clear_cooldown:
                self.canvas[:] = 0
                self.strokes.clear()
                self.redo_stack.clear()
                self.prev_point = None
                self.last_clear_time = now

        # UNDO
        elif gesture == "left_click" and self.strokes:
            self.redo_stack.append(self.strokes.pop())
            self._redraw_all()
            self.prev_point = None

        # REDO
        elif gesture == "right_click" and self.redo_stack:
            self.strokes.append(self.redo_stack.pop())
            self._redraw_all()
            self.prev_point = None

        else:
            self.prev_point = None

        self.last_gesture = gesture

    def _redraw_all(self):
        self.canvas[:] = 0
        for stroke in self.strokes:
            pts = stroke["pts"]
            for i in range(len(pts) - 1):
                cv2.line(
                    self.canvas,
                    pts[i], pts[i + 1],
                    stroke["color"], stroke["thick"]
                )
