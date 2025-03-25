import numpy as np
import tensorflow as tf
import pickle

class GestureClassifier:
    """
    Single TFLite-based classifier that handles ALL gestures (cursor_move, clicks, scroll,
    presentation, teaching, etc.).
    """
    def __init__(self,
                 tflite_path="gesture_model.tflite",
                 encoder_path="gesture_label_encoder.pkl"):
        # Load the label encoder
        with open(encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        # Load the TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict_gesture(self, landmarks):
        """
        Takes a list/array of 42 features (21 x,y).
        Returns the gesture name (string).
        """
        features = np.array(landmarks, dtype=np.float32).reshape(1, -1)
        self.interpreter.set_tensor(self.input_details[0]['index'], features)
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])  # shape (1, num_classes)
        class_index = np.argmax(predictions)
        gesture_name = self.label_encoder.inverse_transform([class_index])[0]
        return gesture_name
