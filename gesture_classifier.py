import os
import sys
import numpy as np
import tensorflow as tf
import pickle

class GestureClassifier:
    """
    Loads the TFLite gesture model and label encoder from bundled files,
    supporting both normal execution and PyInstaller one‐file/one‐folder distributions.
    """
    def __init__(self,
                 tflite_path="gesture_model.tflite",
                 encoder_path="gesture_label_encoder.pkl"):

        # 1) Determine base path for bundled data:
        #    - When frozen by PyInstaller: sys._MEIPASS points to the temp extraction folder.
        #    - Otherwise, __file__'s directory is your source directory.
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            base_dir = sys._MEIPASS
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))

        # 2) Construct absolute paths to your data files:
        model_file   = os.path.join(base_dir, tflite_path)
        encoder_file = os.path.join(base_dir, encoder_path)

        # 3) Load label encoder
        if not os.path.exists(encoder_file):
            raise FileNotFoundError(f"Label encoder not found at {encoder_file}")
        with open(encoder_file, "rb") as f:
            self.label_encoder = pickle.load(f)

        # 4) Load TFLite model
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"TFLite model not found at {model_file}")
        self.interpreter = tf.lite.Interpreter(model_path=model_file)
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict_gesture(self, landmarks):
        features = np.array(landmarks, dtype=np.float32).reshape(1, -1)
        self.interpreter.set_tensor(self.input_details[0]['index'], features)
        self.interpreter.invoke()
        preds = self.interpreter.get_tensor(self.output_details[0]['index'])
        class_idx    = np.argmax(preds)
        gesture_name = self.label_encoder.inverse_transform([class_idx])[0]
        return gesture_name
