#train_model.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# 1. Load the Dataset
#  - Contains all gestures (cursor_move, left_click, right_click, scroll_up, scroll_down,
#    slide_next, draw, erase, etc.), each row = 42 features + 1 label
data = pd.read_csv("gestures.csv")  # Adjust path as needed

# 2. Separate Features & Labels
X = data.iloc[:, :-1].values  # shape (num_samples, 42)
y = data.iloc[:, -1].values   # shape (num_samples,)

# 3. Encode String Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# 5. Define MLP Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(42,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 6. Train
model.fit(X_train, y_train, epochs=25, batch_size=8, validation_split=0.2)

# 7. Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")

# 8. Save Keras Model & Label Encoder
model.save("gesture_model.h5")
with open("gesture_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Training complete. 'gesture_model.h5' + 'gesture_label_encoder.pkl' created.")
