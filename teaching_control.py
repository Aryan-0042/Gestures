import cv2
import mediapipe as mp
import numpy as np
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

        # We keep a separate "canvas" for drawing, so lines persist across frames
        self.canvas = None

        # Track the last gesture recognized
        self.last_gesture = None

        # For freehand drawing or erasing, we store strokes:
        # Each stroke = {"color": (r,g,b), "points": [(x1,y1), (x2,y2), ...]}
        self.strokes = []
        self.redo_stack = []  # For reapplying undone strokes

        # For pointer (ephemeral), we just draw a circle each frame, no persistence
        self.pointer_active = False
        self.pointer_pos = None

        # Keep track of the last fingertip coords for drawing lines
        self.prev_x = None
        self.prev_y = None

        # Canvas background color (black)
        self.bg_color = (0, 0, 0)

        # Colors for draw vs. erase
        self.draw_color = (255, 255, 255)   # white
        self.erase_color = (0, 0, 0)       # black

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
            h, w, _ = frame.shape

            # Initialize canvas if None
            if self.canvas is None:
                self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
                self.canvas[:] = self.bg_color  # Fill with black

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb_frame)

            gesture = "no_gesture"

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # (Optional) draw hand landmarks
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                    # Extract 42 features
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append(lm.x)
                        landmarks.append(lm.y)

                    # Classify gesture
                    gesture = self.classifier.predict_gesture(landmarks)
                    self.perform_action(gesture, frame, hand_landmarks)
            else:
                # No hand detected
                self.perform_action(gesture, frame, None)

            # Draw all strokes onto self.canvas
            self.redraw_strokes()

            # Combine the canvas with the live camera feed for display
            display_frame = cv2.addWeighted(frame, 0.5, self.canvas, 0.5, 0)

            # If pointer is active, draw ephemeral circle on display_frame
            if self.pointer_active and self.pointer_pos:
                cv2.circle(display_frame, self.pointer_pos, 10, (0, 255, 0), 2)

            # Overlay recognized gesture text
            gesture_text = f"Gesture: {gesture}"
            cv2.putText(
                display_frame,
                gesture_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("Teaching Mode (Advanced)", display_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                self.stop()

        cap.release()
        cv2.destroyAllWindows()

    def perform_action(self, gesture, frame, hand_landmarks):
        """
        Gestures:
          - "draw": freehand lines
          - "erase": black lines
          - "clear_canvas": reset board
          - "pointer": ephemeral highlight
          - "undo": revert last stroke
          - "redo": reapply undone stroke
          - "no_gesture": do nothing
        """

        # If we changed from draw/erase to something else, finalize the stroke
        if self.last_gesture in ["draw", "erase"] and gesture not in ["draw", "erase"]:
            self.prev_x = None
            self.prev_y = None

        if gesture == "draw" and hand_landmarks:
            # Freedraw lines in white
            h, w, _ = frame.shape
            index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)

            # If we just started drawing or changed from something else
            if self.last_gesture != "draw":
                # Start a new stroke
                self.strokes.append({"color": self.draw_color, "points": []})
                self.redo_stack.clear()  # new action invalidates redo stack

            # Add current point to the last stroke
            self.strokes[-1]["points"].append((cx, cy))

            # If we have a previous point, we can connect them
            if self.prev_x is not None and self.prev_y is not None:
                cv2.line(self.canvas, (self.prev_x, self.prev_y), (cx, cy), self.draw_color, 3)

            self.prev_x, self.prev_y = cx, cy
            self.pointer_active = False  # pointer off

        elif gesture == "erase" and hand_landmarks:
            # Freedraw lines in black
            h, w, _ = frame.shape
            index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)

            if self.last_gesture != "erase":
                # Start a new stroke (black)
                self.strokes.append({"color": self.erase_color, "points": []})
                self.redo_stack.clear()

            self.strokes[-1]["points"].append((cx, cy))

            if self.prev_x is not None and self.prev_y is not None:
                cv2.line(self.canvas, (self.prev_x, self.prev_y), (cx, cy), self.erase_color, 20)
                # Thicker line for erasing

            self.prev_x, self.prev_y = cx, cy
            self.pointer_active = False

        elif gesture == "clear_canvas" and self.last_gesture != "clear_canvas":
            # Reset board
            self.canvas[:] = self.bg_color
            self.strokes.clear()
            self.redo_stack.clear()
            self.prev_x = None
            self.prev_y = None
            self.pointer_active = False

        elif gesture == "pointer" and hand_landmarks:
            # Just show a circle at fingertip, do not persist
            h, w, _ = frame.shape
            index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            self.pointer_active = True
            self.pointer_pos = (cx, cy)

            # If we were drawing/erasing, finalize that stroke
            if self.last_gesture in ["draw", "erase"]:
                self.prev_x = None
                self.prev_y = None

        elif gesture == "undo" and self.last_gesture != "undo":
            # Remove last stroke from self.strokes, push onto redo_stack
            if self.strokes:
                last_stroke = self.strokes.pop()
                self.redo_stack.append(last_stroke)
                # Redraw everything
                self.redraw_canvas()
            self.pointer_active = False

        elif gesture == "redo" and self.last_gesture != "redo":
            # Reapply last undone stroke
            if self.redo_stack:
                stroke = self.redo_stack.pop()
                self.strokes.append(stroke)
                self.redraw_canvas()
            self.pointer_active = False

        else:
            # no_gesture or unrecognized
            self.pointer_active = False
            self.prev_x = None
            self.prev_y = None

        self.last_gesture = gesture

    def redraw_canvas(self):
        """
        Clears self.canvas and redraws all strokes from scratch.
        Useful after undo/redo or clear.
        """
        if self.canvas is None:
            return
        self.canvas[:] = self.bg_color  # reset to black

        # Replay each stroke
        for stroke in self.strokes:
            color = stroke["color"]
            points = stroke["points"]
            if len(points) < 2:
                continue
            # Draw line segments
            for i in range(len(points) - 1):
                cv2.line(self.canvas, points[i], points[i+1], color, 3 if color != (0,0,0) else 20)

    def redraw_strokes(self):
        """
        Called each frame to ensure any new partial lines are displayed.
        If you don't need partial lines, you can rely on redraw_canvas() only after finalizing strokes.
        """
        # In this approach, we do partial lines directly in perform_action()
        # so there's nothing special to do here unless you want partial logic.
        pass

