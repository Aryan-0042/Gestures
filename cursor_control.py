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

        # Track last recognized gesture
        self.last_gesture = None

        # Cursor movement smoothing
        self.smooth_x = None
        self.smooth_y = None
        self.alpha = 0.2  # smoothing factor for cursor

        # Velocity-based scrolling
        self.scroll_velocity = 0.0     # current scroll velocity
        self.scroll_accel = 1.4        # how fast velocity changes each frame (~40% more than 1.0)
        self.scroll_damp = 0.8        # damping factor when no scroll gesture
        self.scroll_min_threshold = 0.5  # if abs(velocity) < threshold, set to 0

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

            # Apply velocity-based scrolling
            self.apply_smooth_scrolling()

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

            cv2.imshow("Gesture Control (Smooth Scroll)", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                self.stop()

        cap.release()
        cv2.destroyAllWindows()

    def perform_action(self, gesture, hand_landmarks):
        """
        Handles gestures:
         - cursor_move, left_click, right_click, double_click
         - scroll_up, scroll_down
         - Possibly other gestures like slide_next, zoom_in, etc.
         - 'hold_click' has been removed entirely.
        """

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

        elif gesture == "no_gesture":
            pass

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
            self.scroll_velocity += self.scroll_accel  # accelerate upward

        elif gesture == "scroll_down":
            self.scroll_velocity -= self.scroll_accel  # accelerate downward

        # Additional gestures for presentation/teaching if needed
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

        self.last_gesture = gesture

    def apply_smooth_scrolling(self):
        """
        Velocity-based scrolling with damping for a smoother feel.
        """
        # Dampen velocity if no active scroll gesture
        self.scroll_velocity *= self.scroll_damp

        # If velocity is small, zero it out
        if abs(self.scroll_velocity) < self.scroll_min_threshold:
            self.scroll_velocity = 0

        # If velocity is >= 1 or <= -1, apply it
        if abs(self.scroll_velocity) >= 1:
            pyautogui.scroll(int(self.scroll_velocity))
