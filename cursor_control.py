import cv2
import mediapipe as mp
import pyautogui
import threading
import time
from gesture_classifier import GestureClassifier

class VideoStream:
    """
    Threaded video capture class that continuously reads frames.
    """
    def __init__(self, src=0):  # Using default camera (index 0)
        self.stream = cv2.VideoCapture(src)
        # Optionally lower resolution for faster processing
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()
            if not self.grabbed:
                print("Warning: Frame not grabbed. Retrying...")
                time.sleep(0.01)
                continue

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

class CursorControl:
    def __init__(self):
        # Instantiate two classifiers:
        # Lite classifier for cursor movement.
        self.cursor_classifier = GestureClassifier(mode="lite")
        # Default classifier for non-cursor gestures.
        self.default_classifier = GestureClassifier(mode="default")
        
        self.mp_hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.screen_width, self.screen_height = pyautogui.size()
        self.running = False

        # Track "hold click" state and other variables
        self.is_holding = False
        self.last_gesture = None
        self.smooth_x = None
        self.smooth_y = None
        self.alpha = 0.2  # Smoothing factor
        self.scroll_count = 0         # For scroll gesture handling
        self.max_scroll_speed = 20    # Caps the scroll speed

    def start(self):
        self.running = True
        self.run_cursor_control()

    def stop(self):
        self.running = False

    def run_cursor_control(self):
        vs = VideoStream().start() 
        #vs = VideoStream(src=0).start() # Using default camera
        while self.running:
            frame = vs.read()
            if frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb_frame)
            gesture = "no_gesture"  # Default

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Optionally, draw hand landmarks for visualization
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS
                    )
                    # Extract features (21 x,y pairs)
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append(lm.x)
                        landmarks.append(lm.y)
                    
                    # First, use the lite classifier
                    lite_gesture = self.cursor_classifier.predict_gesture(landmarks)
                    if lite_gesture == "cursor_move":
                        gesture = lite_gesture
                    else:
                        # For non-cursor gestures, use the default classifier
                        gesture = self.default_classifier.predict_gesture(landmarks)
                    
                    self.perform_action(gesture, hand_landmarks)
            else:
                self.perform_action(gesture, None)

            # Show gesture information on screen
            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            cv2.imshow("Cursor Mode (Dual Classifier)", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                self.stop()

        vs.stop()
        cv2.destroyAllWindows()

    def perform_action(self, gesture, hand_landmarks):
        """
        Executes actions based on the recognized gesture.
        """
        # 1. If we were holding click and it's no longer hold_click, release mouse.
        if self.last_gesture == "hold_click" and gesture != "hold_click":
            pyautogui.mouseUp()
            self.is_holding = False

        # 2. Handle cursor movement.
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

        else:
            if self.is_holding:
                pyautogui.mouseUp()
                self.is_holding = False
            self.scroll_count = 0

        self.last_gesture = gesture

if __name__ == '__main__':
    cc = CursorControl()
    cc.start()
