import cv2
import mediapipe as mp
import pyautogui
import time
from gesture_classifier import GestureClassifier

class PresentationControl:
    def __init__(self):
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

        # Debounce trackers (timestamps)
        self.last_next = 0      # for next_slide (3s)
        self.last_prev = 0      # for prev_slide (3s)
        self.last_zoom = 0      # for zoom in/out (3s)
        self.last_exit = 0      # for exit slideshow (3s)

        # Debounce intervals in seconds
        self.INTERVAL_SLIDE = 3.0
        self.INTERVAL_ZOOM  = 3.0
        self.INTERVAL_EXIT  = 3.0

        self.zoom_counter = 0
        self.max_zoom = 5

        self.prev_raw = "no_gesture"
        self.gesture_cooldown = 0.1  # 100 ms grace to filter flicker

    def process_frame(self, frame, overlay=True):
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb)

        raw = "no_gesture"
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            if overlay:
                self.mp_draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            # extract features...
            feats = []
            for lm in hand.landmark:
                feats += [lm.x, lm.y]
            raw = self.classifier.predict_gesture(feats)

        # If switching back to no_gesture, start a short flicker timer
        now = time.time()
        if raw == "no_gesture" and self.prev_raw != "no_gesture":
            self._no_start = now

        # Only accept the new gesture if it’s been stable for >= cooldown
        if raw != self.prev_raw or raw == "no_gesture":
            # flicker guard
            if hasattr(self, "_no_start") and raw == "no_gesture":
                if now - self._no_start < self.gesture_cooldown:
                    raw = self.prev_raw  # hold old
            # else accept
        # Else raw==prev_raw: steady gesture

        # Perform the action
        self.perform_action(raw, now)

        # overlay text
        if overlay:
            cv2.putText(frame, raw.replace("_", " ").title(), (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.putText(frame, f"Zoom: {self.zoom_counter}", (10,80),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

        self.prev_raw = raw
        return frame

    def perform_action(self, gesture, now=None):
        if now is None:
            now = time.time()

        # NEXT SLIDE
        if gesture == "next_slide":
            if now - self.last_next >= self.INTERVAL_SLIDE:
                pyautogui.press("right")
                self.last_next = now

        # PREVIOUS SLIDE
        elif gesture == "prev_slide":
            if now - self.last_prev >= self.INTERVAL_SLIDE:
                pyautogui.press("left")
                self.last_prev = now

        # ZOOM IN  (scroll_down)
        elif gesture == "scroll_down" and self.zoom_counter < self.max_zoom:
            if now - self.last_zoom >= self.INTERVAL_ZOOM:
                pyautogui.hotkey("ctrl", "+")
                self.zoom_counter += 1
                self.last_zoom = now

        # ZOOM OUT (scroll_up)
        elif gesture == "scroll_up" and self.zoom_counter > 0:
            if now - self.last_zoom >= self.INTERVAL_ZOOM:
                pyautogui.hotkey("ctrl", "-")
                self.zoom_counter -= 1
                self.last_zoom = now

        # START SLIDESHOW
        elif gesture == "double_click":
            pyautogui.press("f5")

        # EXIT SLIDESHOW
        elif gesture == "left_click":
            if now - self.last_exit >= self.INTERVAL_EXIT:
                pyautogui.press("esc")
                self.last_exit = now

        # no action on “no_gesture”
        # any other predicted gesture → ignored
