import cv2
import mediapipe as mp
from gesture_classifier import GestureClassifier

class TeachingControl:
    def __init__(self):
        self.classifier = GestureClassifier()
        self.mp_hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.running = False

        self.canvas = None
        self.last_gesture = None

    def start(self):
        self.running = True
        self.run_teaching()

    def stop(self):
        self.running = False

    def run_teaching(self):
        cap = cv2.VideoCapture(0)

        while self.running:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            if self.canvas is None:
                self.canvas = frame.copy()  # Initialize canvas

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb_frame)

            gesture = "no_gesture"

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                    # Extract 42 features
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append(lm.x)
                        landmarks.append(lm.y)

                    gesture = self.classifier.predict_gesture(landmarks)
                    self.perform_action(gesture, frame, hand_landmarks)
            else:
                self.perform_action(gesture, frame, None)

            # === Overlay recognized gesture ===
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

            cv2.imshow("Teaching Mode (ANN + Feedback)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                self.stop()

        cap.release()
        cv2.destroyAllWindows()

    def perform_action(self, gesture, frame, hand_landmarks):
        """
        Example gestures: "draw", "erase", "clear_canvas"
        We do them once or continuously depending on how we define them.
        """
        if gesture == "draw" and hand_landmarks:
            # Example: draw a small white circle at index fingertip
            h, w, _ = frame.shape
            index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)

        elif gesture == "erase" and hand_landmarks:
            # Draw black circle to erase
            h, w, _ = frame.shape
            index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 0), -1)

        elif gesture == "clear_canvas" and self.last_gesture != "clear_canvas":
            # Fill frame with black
            frame[:] = (0, 0, 0)

        # You can define more logic if needed (one-time triggers, etc.)

        self.last_gesture = gesture
