import cv2
import mediapipe as mp
import pyautogui
from gesture_classifier import GestureClassifier

class PresentationControl:
    def __init__(self):
        self.classifier = GestureClassifier()
        self.mp_hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.running = False

        # Track last recognized gesture
        self.last_gesture = None

        # For pointer gesture: store if active + fingertip coords
        self.pointer_active = False
        self.pointer_pos = None

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
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb_frame)

            gesture = "no_gesture"  # Default if no hand is detected

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # (Optional) draw landmarks
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS
                    )

                    # Extract 42 features (21 x,y pairs)
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append(lm.x)
                        landmarks.append(lm.y)

                    # Predict gesture
                    gesture = self.classifier.predict_gesture(landmarks)
                    self.perform_action(gesture, frame, hand_landmarks)
            else:
                # No hand detected
                self.perform_action(gesture, frame, None)

            # If pointer is active, draw a temporary circle at pointer_pos
            if self.pointer_active and self.pointer_pos:
                cv2.circle(frame, self.pointer_pos, 10, (0, 255, 0), 2)

            # Real-Time Gesture Feedback
            gesture_text = f"Gesture: {gesture}"
            cv2.putText(
                frame,
                gesture_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("Presentation Mode (ANN + Pointer)", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                self.stop()

        cap.release()
        cv2.destroyAllWindows()

    def perform_action(self, gesture, frame, hand_landmarks):
        """
        Handle presentation gestures:
          - next_slide, prev_slide
          - zoom_in, zoom_out
          - pointer (show a green circle on fingertip)
          - start_slideshow, end_slideshow
          - no_gesture
        """
        # If we had pointer active and now gesture changed, disable pointer
        if self.last_gesture == "pointer" and gesture != "pointer":
            self.pointer_active = False
            self.pointer_pos = None

        if gesture == "next_slide" and self.last_gesture != "next_slide":
            pyautogui.press("right")

        elif gesture == "prev_slide" and self.last_gesture != "prev_slide":
            pyautogui.press("left")

        elif gesture == "zoom_in" and self.last_gesture != "zoom_in":
            pyautogui.hotkey("ctrl", "+")

        elif gesture == "zoom_out" and self.last_gesture != "zoom_out":
            pyautogui.hotkey("ctrl", "-")

        elif gesture == "pointer" and hand_landmarks:
            # Just display a circle on fingertip each frame
            h, w, _ = frame.shape
            index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            self.pointer_active = True
            self.pointer_pos = (cx, cy)

        elif gesture == "start_slideshow" and self.last_gesture != "start_slideshow":
            # For PowerPoint: F5 often starts the slideshow
            pyautogui.press("f5")

        elif gesture == "end_slideshow" and self.last_gesture != "end_slideshow":
            # Press ESC to end the slideshow
            pyautogui.press("esc")

        elif gesture == "no_gesture":
            pass

        # Update last gesture
        self.last_gesture = gesture
