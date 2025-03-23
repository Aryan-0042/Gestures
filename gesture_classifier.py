# gesture_classifier.py
import numpy as np
import tensorflow as tf
import pickle

class GestureClassifier:
    def __init__(self, mode="lite",
                 model_path_lite="gesture_model.tflite",
                 model_path_default="gesture_model.h5",
                 encoder_path="gesture_label_encoder.pkl"):
        """
        mode: "lite" uses the TFLite model; "default" uses the full Keras model.
        """
        self.mode = mode
        with open(encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)
        if mode == "lite":
            self.interpreter = tf.lite.Interpreter(model_path=model_path_lite)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            self.model = tf.keras.models.load_model(model_path_default)

    def predict_gesture(self, landmarks):
        features = np.array(landmarks, dtype=np.float32).reshape(1, -1)
        if self.mode == "lite":
            self.interpreter.set_tensor(self.input_details[0]['index'], features)
            self.interpreter.invoke()
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
        else:
            predictions = self.model.predict(features)
        class_index = np.argmax(predictions)
        gesture_name = self.label_encoder.inverse_transform([class_index])[0]
        return gesture_name
