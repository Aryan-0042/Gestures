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
        self.last_gesture = None
        self.zoom_counter = 0
        self.max_zoom = 5

        # Timestamps for delayed actions:
        self.last_slide_time = 0  # For next_slide and prev_slide (3 sec delay)
        self.last_zoom_time = 0   # For scroll_down and scroll_up (2 sec delay)

    def process_frame(self, frame, overlay=True):
        """
        Process a single frame for presentation control.
        - Flips the frame.
        - Uses MediaPipe to detect hand landmarks.
        - Classifies the gesture and maps it to a control action (with delay).
        - Overlays the gesture and zoom level (if overlay is enabled).
        Returns the processed frame.
        """
        # Flip the frame to mirror the user's view.
        frame = cv2.flip(frame, 1)

        # Convert frame for MediaPipe processing.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb_frame)

        gesture = "no_gesture"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if overlay:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS
                    )
                # Extract 42 features: 21 x,y pairs.
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                gesture = self.classifier.predict_gesture(landmarks)
                self.perform_action(gesture)
                break  # Process only the first detected hand.
        else:
            self.perform_action("no_gesture")

        if overlay:
            cv2.putText(frame, f"Gesture: {gesture}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Zoom Level: {self.zoom_counter}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        return frame

    def perform_action(self, gesture):
        """
        Maps predicted gestures to presentation commands:
          - "left_click": Mouse click.
          - "right_click": Mouse right-click.
          - "next_slide": Advance to the next slide (3-sec delay).
          - "prev_slide": Go back to the previous slide (3-sec delay).
          - "scroll_down": Zoom in (if zoom level < max) with a 2-sec delay.
          - "scroll_up": Zoom out (if zoom level > 0) with a 2-sec delay.
          - "double_click": Start slideshow.
          - "end_slideshow": End slideshow.
        """
        current_time = time.time()

        if gesture == "left_click" and self.last_gesture != "left_click":
            pyautogui.click()
        elif gesture == "right_click" and self.last_gesture != "right_click":
            pyautogui.rightClick()
        elif gesture == "next_slide" and self.last_gesture != "next_slide":
            # Only process if 3 seconds have elapsed.
            if current_time - self.last_slide_time >= 3:
                pyautogui.press("right")
                self.last_slide_time = current_time
        elif gesture == "prev_slide" and self.last_gesture != "prev_slide":
            if current_time - self.last_slide_time >= 3:
                pyautogui.press("left")
                self.last_slide_time = current_time
        elif gesture == "scroll_down" and self.zoom_counter < self.max_zoom:
            if current_time - self.last_zoom_time >= 2:
                pyautogui.hotkey("ctrl", "+")
                self.zoom_counter += 1
                self.last_zoom_time = current_time
        elif gesture == "scroll_up" and self.zoom_counter > 0:
            if current_time - self.last_zoom_time >= 2:
                pyautogui.hotkey("ctrl", "-")
                self.zoom_counter -= 1
                self.last_zoom_time = current_time
        elif gesture == "double_click" and self.last_gesture != "double_click":
            pyautogui.press("f5")
        elif gesture == "end_slideshow" and self.last_gesture != "end_slideshow":
            pyautogui.press("esc")

        self.last_gesture = gesture
