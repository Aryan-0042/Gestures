# cursor_control.py
import cv2
import mediapipe as mp
import pyautogui
from gesture_classifier import GestureClassifier

class CursorControl:
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
        self.hand_connections = mp.solutions.hands.HAND_CONNECTIONS

        self.screen_width, self.screen_height = pyautogui.size()
        self.last_gesture = None
        self.smooth_x = None
        self.smooth_y = None
        self.alpha = 0.2

        self.scroll_velocity = 0.0
        self.scroll_accel = 1.4
        self.scroll_damp = 0.8
        self.scroll_min_threshold = 0.5

    def process_frame(self, frame, overlay=True):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb_frame)

        gesture = "no_gesture"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if overlay:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.hand_connections)

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)

                gesture = self.classifier.predict_gesture(landmarks)
                self.perform_action(gesture, hand_landmarks)
        else:
            self.perform_action("no_gesture", None)

        self.apply_smooth_scrolling()

        if overlay:
            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return frame  # return processed frame for UI display

    def perform_action(self, gesture, hand_landmarks):
        if gesture == "cursor_move" and hand_landmarks:
            index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            x_new = int(index_tip.x * self.screen_width)
            y_new = int(index_tip.y * self.screen_height)
            if self.smooth_x is None or self.smooth_y is None:
                self.smooth_x = x_new
                self.smooth_y = y_new
            else:
                self.smooth_x = self.alpha * x_new + (1 - self.alpha) * self.smooth_x
                self.smooth_y = self.alpha * y_new + (1 - self.alpha) * self.smooth_y
            pyautogui.moveTo(int(self.smooth_x), int(self.smooth_y))

        elif gesture == "left_click":
            if self.last_gesture != "left_click":
                pyautogui.click()
        elif gesture == "right_click":
            if self.last_gesture != "right_click":
                pyautogui.rightClick()
        elif gesture == "double_click":
            if self.last_gesture != "double_click":
                pyautogui.doubleClick()
        elif gesture == "scroll_up":
            self.scroll_velocity += self.scroll_accel
        elif gesture == "scroll_down":
            self.scroll_velocity -= self.scroll_accel
        elif gesture == "slide_next":
            pyautogui.press("right")
        elif gesture == "slide_prev":
            pyautogui.press("left")
        elif gesture == "zoom_in":
            pyautogui.hotkey("ctrl", "+")
        elif gesture == "zoom_out":
            pyautogui.hotkey("ctrl", "-")

        self.last_gesture = gesture

    def apply_smooth_scrolling(self):
        self.scroll_velocity *= self.scroll_damp
        if abs(self.scroll_velocity) < self.scroll_min_threshold:
            self.scroll_velocity = 0
        if abs(self.scroll_velocity) >= 1:
            pyautogui.scroll(int(self.scroll_velocity))
