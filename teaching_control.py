import cv2
import mediapipe as mp
import numpy as np
import time
from gesture_classifier import GestureClassifier

class TeachingControl:
    def __init__(self, show_gesture=True):
        # Load the unified TFLite gesture classifier.
        self.classifier = GestureClassifier(
            tflite_path="gesture_model.tflite",
            encoder_path="gesture_label_encoder.pkl"
        )
        # Initialize MediaPipe Hands.
        self.mp_hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.running = False

        # Persistent whiteboard canvas (initialized on first frame)
        self.canvas = None
        
        # Data structures for stroke management.
        self.strokes = []       # Each stroke is a dict: {"points": [list of points], "color": (B, G, R), "thickness": int}
        self.redo_stack = []    # For storing undone strokes.

        # Drawing parameters.
        self.draw_color = (255, 255, 255)  # White
        self.erase_color = (0, 0, 0)         # Black
        self.draw_thickness = 3
        self.erase_thickness = 30          # Increased eraser thickness

        self.eraser_on = False  # Toggle for eraser mode (can be set externally)

        # For continuous drawing, store the previous fingertip position.
        self.prev_point = None
        self.last_gesture = "no_gesture"
        self.show_gesture = show_gesture

        # Cooldown for clearing the canvas (mapped to "double_click")
        self.last_clear_time = 0
        self.clear_cooldown = 2  # seconds

    def process_frame(self, frame, overlay=True):
        """
        Processes an input frame for teaching mode.
        - Flips and converts colors.
        - Initializes the whiteboard canvas if necessary.
        - Processes hand landmarks with MediaPipe.
        - Predicts gesture and executes whiteboard actions.
        - Overlays the gesture label (if enabled) onto the blended output.
        Returns the resulting frame.
        """
        # Flip for mirror effect.
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Initialize or update the canvas to match the frame dimensions.
        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
            self.canvas[:] = (0, 0, 0)

        # Convert the frame to RGB for hand landmark processing.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb_frame)

        gesture = "no_gesture"
        if results.multi_hand_landmarks:
            # Process only the first detected hand.
            hand_landmarks = results.multi_hand_landmarks[0]
            # Optionally draw landmarks on the frame.
            if overlay:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            # Extract 42 features (21 x,y pairs) from hand landmarks.
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])
            # Predict the gesture.
            gesture = self.classifier.predict_gesture(landmarks)
            self.perform_action(gesture, frame, hand_landmarks)
        else:
            self.perform_action("no_gesture", frame, None)

        # Blend the live frame with the whiteboard canvas.
        overlay_frame = cv2.addWeighted(frame, 0.5, self.canvas, 0.5, 0)
        if overlay and self.show_gesture:
            cv2.putText(overlay_frame, f"Gesture: {gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return overlay_frame

    def perform_action(self, gesture, frame, hand_landmarks):
        """
        Maps the predicted gesture to a whiteboard action:
         - "cursor_move": Draw or erase freehand.
         - "double_click": Clear the canvas (with a 2-sec cooldown).
         - "left_click": Undo the last stroke.
         - "right_click": Redo the last undone stroke.
         - Otherwise: Reset the drawing pointer.
        """
        h, w, _ = frame.shape
        now = time.time()
        current_point = None
        if hand_landmarks:
            index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            current_point = (int(index_tip.x * w), int(index_tip.y * h))

        if gesture == "cursor_move":
            if self.last_gesture != "cursor_move":
                color = self.erase_color if self.eraser_on else self.draw_color
                thickness = self.erase_thickness if self.eraser_on else self.draw_thickness
                self.strokes.append({"points": [], "color": color, "thickness": thickness})
                self.redo_stack.clear()
                self.prev_point = None
            if current_point:
                self.strokes[-1]["points"].append(current_point)
                if self.prev_point is not None:
                    col = self.erase_color if self.eraser_on else self.draw_color
                    thick = self.erase_thickness if self.eraser_on else self.draw_thickness
                    cv2.line(self.canvas, self.prev_point, current_point, col, thick)
                self.prev_point = current_point

        elif gesture == "double_click":
            # Clear the canvas with a 2-second cooldown.
            if self.last_gesture != "double_click" and (now - self.last_clear_time) > self.clear_cooldown:
                self.canvas[:] = (0, 0, 0)
                self.strokes.clear()
                self.redo_stack.clear()
                self.prev_point = None
                self.last_clear_time = now

        elif gesture == "left_click":
            # Undo the last stroke.
            if self.last_gesture != "left_click" and self.strokes:
                self.redo_stack.append(self.strokes.pop())
                self.redraw_canvas()
                self.prev_point = None

        elif gesture == "right_click":
            # Redo the last undone stroke.
            if self.last_gesture != "right_click" and self.redo_stack:
                self.strokes.append(self.redo_stack.pop())
                self.redraw_canvas()
                self.prev_point = None

        else:
            self.prev_point = None

        self.last_gesture = gesture

    def redraw_canvas(self):
        """Clears and redraws all strokes onto the canvas."""
        if self.canvas is None:
            return
        self.canvas[:] = (0, 0, 0)
        for stroke in self.strokes:
            pts = stroke["points"]
            col = stroke["color"]
            thick = stroke["thickness"]
            if len(pts) > 1:
                for i in range(len(pts)-1):
                    cv2.line(self.canvas, pts[i], pts[i+1], col, thick)

    def get_color_name(self):
        """Returns a human-friendly name for the current drawing color."""
        names = {
            (255, 255, 255): "White",
            (0, 0, 255): "Red",
            (255, 0, 0): "Blue",
            (0, 255, 0): "Green"
        }
        return names.get(self.draw_color, "Custom")
