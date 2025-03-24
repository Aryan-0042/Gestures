import tensorflow as tf

# Load your Keras model
model = tf.keras.models.load_model("gesture_model.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Optional: enable optimization for smaller/faster TFLite
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save TFLite model
with open("gesture_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as 'gesture_model.tflite'.")
