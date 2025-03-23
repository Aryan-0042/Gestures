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
                    # Draw landmarks (optional)
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
                    self.perform_action(gesture)

            else:
                # Handle transitions if needed
                self.perform_action(gesture)

            # === Real-Time Gesture Feedback Overlay ===
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

            cv2.imshow("Presentation Mode (ANN + Feedback)", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                self.stop()

        cap.release()
        cv2.destroyAllWindows()

    def perform_action(self, gesture):
        """
        Handle presentation gestures (next_slide, prev_slide, zoom_in, etc.)
        One-time triggers for each gesture if it differs from self.last_gesture.
        """
        if gesture == "next_slide" and self.last_gesture != "next_slide":
            pyautogui.press("right")

        elif gesture == "prev_slide" and self.last_gesture != "prev_slide":
            pyautogui.press("left")

        elif gesture == "zoom_in" and self.last_gesture != "zoom_in":
            pyautogui.hotkey("ctrl", "+")

        elif gesture == "zoom_out" and self.last_gesture != "zoom_out":
            pyautogui.hotkey("ctrl", "-")

        # Add more presentation gestures as needed

        # Update last gesture
        self.last_gesture = gesture
