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
        self.running = False
        self.last_gesture = None
        self.zoom_counter = 0
        self.max_zoom = 5

    def start(self):
        self.running = True
        self.run_presentation()

    def stop(self):
        self.running = False

    def run_presentation(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            success, frame = cap.read()
            if not success:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb_frame)

            gesture = "no_gesture"
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS
                    )
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append(lm.x)
                        landmarks.append(lm.y)
                    gesture = self.classifier.predict_gesture(landmarks)
                    self.perform_action(gesture)
            else:
                self.perform_action(gesture)

            cv2.putText(frame, f"Gesture: {gesture}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Zoom Level: {self.zoom_counter}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Presentation Mode", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                self.stop()

        cap.release()
        cv2.destroyAllWindows()

    def perform_action(self, gesture):
        if gesture == "left_click" and self.last_gesture != "left_click":
            pyautogui.click()
        elif gesture == "right_click" and self.last_gesture != "right_click":
            pyautogui.rightClick()
        elif gesture == "next_slide" and self.last_gesture != "next_slide":
            pyautogui.press("right")
        elif gesture == "prev_slide" and self.last_gesture != "prev_slide":
            pyautogui.press("left")
        elif gesture == "scroll_down" and self.zoom_counter < self.max_zoom:
            pyautogui.hotkey("ctrl", "+")
            self.zoom_counter += 1
        elif gesture == "scroll_up" and self.zoom_counter > 0:
            pyautogui.hotkey("ctrl", "-")
            self.zoom_counter -= 1
        elif gesture == "double_click" and self.last_gesture != "double_click":
            pyautogui.press("f5")
        elif gesture == "end_slideshow" and self.last_gesture != "end_slideshow":
            pyautogui.press("esc")
        self.last_gesture = gesture
