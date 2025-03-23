import tensorflow as tf

# Load your existing model
model = tf.keras.models.load_model("gesture_model.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optional: enable optimizations
tflite_model = converter.convert()

# Save the TFLite model
with open("gesture_model.tflite", "wb") as f:
    f.write(tflite_model)
