import cv2
import mediapipe as mp
import pyautogui
import time
from gesture_classifier import GestureClassifier

class CursorControl:
    def __init__(self):
        # Single TFLite classifier for all gestures
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

        self.screen_width, self.screen_height = pyautogui.size()
        self.running = False

        # Track states
        self.is_holding = False
        self.last_gesture = None
        self.smooth_x = None
        self.smooth_y = None
        self.alpha = 0.2  # smoothing factor

        # Scroll handling
        self.scroll_count = 0
        self.max_scroll_speed = 20

    def start(self):
        self.running = True
        self.run_cursor_control()

    def stop(self):
        self.running = False

    def run_cursor_control(self):
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
                    # Optional: draw landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                    )

                    # Extract 42 features (21 x,y)
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append(lm.x)
                        landmarks.append(lm.y)

                    # Single TFLite classifier for ALL gestures
                    gesture = self.classifier.predict_gesture(landmarks)
                    self.perform_action(gesture, hand_landmarks)
            else:
                self.perform_action(gesture, None)

            # Display recognized gesture
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

            cv2.imshow("Gesture Control (Single TFLite)", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                self.stop()

        cap.release()
        cv2.destroyAllWindows()

    def perform_action(self, gesture, hand_landmarks):
        """
        Handles all gestures: 
         - cursor_move, left_click, right_click, hold_click, scroll_up, scroll_down
         - double_click (newly added)
         - Potential presentation or teaching gestures (slide_next, draw, etc.)
        """

        # If we were holding click and gesture changed, release mouse
        if self.last_gesture == "hold_click" and gesture != "hold_click":
            pyautogui.mouseUp()
            self.is_holding = False

        if gesture == "cursor_move" and hand_landmarks:
            index_tip = hand_landmarks.landmark[
                mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP
            ]
            x_new = int(index_tip.x * self.screen_width)
            y_new = int(index_tip.y * self.screen_height)
            if self.smooth_x is None or self.smooth_y is None:
                self.smooth_x = x_new
                self.smooth_y = y_new
            else:
                self.smooth_x = self.alpha * x_new + (1 - self.alpha) * self.smooth_x
                self.smooth_y = self.alpha * y_new + (1 - self.alpha) * self.smooth_y
            pyautogui.moveTo(int(self.smooth_x), int(self.smooth_y))
            self.scroll_count = 0

        elif gesture == "no_gesture":
            self.scroll_count = 0

        elif gesture == "left_click":
            if self.last_gesture != "left_click":
                pyautogui.click()
            self.scroll_count = 0

        elif gesture == "right_click":
            if self.last_gesture != "right_click":
                pyautogui.rightClick()
            self.scroll_count = 0

        elif gesture == "hold_click":
            if self.last_gesture != "hold_click":
                pyautogui.mouseDown()
                self.is_holding = True
            self.scroll_count = 0

        elif gesture == "scroll_up":
            if self.last_gesture == "scroll_up":
                self.scroll_count += 1
            else:
                self.scroll_count = 1
            scroll_speed = min(self.scroll_count, self.max_scroll_speed)
            pyautogui.scroll(scroll_speed)

        elif gesture == "scroll_down":
            if self.last_gesture == "scroll_down":
                self.scroll_count += 1
            else:
                self.scroll_count = 1
            scroll_speed = min(self.scroll_count, self.max_scroll_speed)
            pyautogui.scroll(-scroll_speed)

        elif gesture == "double_click":
            # NEW GESTURE: Double-click
            if self.last_gesture != "double_click":
                pyautogui.doubleClick()
            self.scroll_count = 0

        # Additional gestures (presentation/teaching) remain the same
        elif gesture == "slide_next":
            pyautogui.press("right")

        elif gesture == "slide_prev":
            pyautogui.press("left")

        elif gesture == "zoom_in":
            pyautogui.hotkey("ctrl", "+")

        elif gesture == "zoom_out":
            pyautogui.hotkey("ctrl", "-")

        elif gesture == "draw":
            pass
        elif gesture == "erase":
            pass
        elif gesture == "clear_canvas":
            pass

        else:
            # If we had a hold_click or other state, release
            if self.is_holding:
                pyautogui.mouseUp()
                self.is_holding = False
            self.scroll_count = 0

        self.last_gesture = gesture


if __name__ == "__main__":
    cc = CursorControl()
    cc.start()
