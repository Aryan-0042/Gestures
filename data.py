import cv2
import mediapipe as mp
import csv
import os
import pandas as pd

# Default gesture name (if user doesn't type a new one)
DEFAULT_GESTURE = "double_click"
OUTPUT_CSV = "gestures.csv"
MAX_SAMPLES = 2000 # Total samples we want for each gesture

def analyze_dataset(csv_file):
    """Shows basic info about the dataset: row count, columns, gesture distribution."""
    if not os.path.exists(csv_file):
        print(f"No dataset found: {csv_file}")
        return

    # Assuming 21 (x,y) pairs => 42 features, plus 1 label => 43 columns total
    col_names = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + ["label"]

    df = pd.read_csv(csv_file, header=None, names=col_names)
    total_rows = len(df)
    total_cols = df.shape[1]

    print("\n--- Dataset Analysis ---")
    print(f"File: {csv_file}")
    print(f"Total rows: {total_rows}")
    print(f"Total columns: {total_cols}")

    # Distribution of gestures
    gesture_counts = df["label"].value_counts()
    print("\nGesture Distribution:")
    print(gesture_counts)
    print("----------------------\n")

def count_existing_samples(gesture_name, csv_file):
    """
    Counts how many rows of 'gesture_name' are already in csv_file.
    Returns the count (0 if file doesn't exist or gesture not found).
    """
    if not os.path.exists(csv_file):
        return 0

    count = 0
    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[-1] == gesture_name:
                count += 1
    return count

def main():
    # 1) Analyze existing dataset first
    analyze_dataset(OUTPUT_CSV)

    # 2) Ask user if they want to change the gesture name (or keep default)
    user_input = input(f"Enter new gesture name (leave blank for '{DEFAULT_GESTURE}'): ").strip()
    if user_input:
        gesture_name = user_input
    else:
        gesture_name = DEFAULT_GESTURE

    # Count how many samples we already have for this gesture
    existing_count = count_existing_samples(gesture_name, OUTPUT_CSV)
    print(f"Currently have {existing_count} samples for '{gesture_name}'.")

    if existing_count >= MAX_SAMPLES:
        print(f"We already have {existing_count} samples, which is >= {MAX_SAMPLES}. No more needed.")
        return  # Exit without collecting

    needed_count = MAX_SAMPLES - existing_count
    print(f"\nWe need {needed_count} more samples to reach {MAX_SAMPLES} total for '{gesture_name}'.")
    print("Press 'q' to stop at any time.\n")

    # 3) Old code logic (MINIMAL CHANGES)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    collected_count = 0  # How many new samples in this session

    with open(OUTPUT_CSV, mode='a', newline='') as f:
        csv_writer = csv.writer(f)

        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read from camera. Exiting.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Extract 42 features (21 x,y)
                    row = []
                    for lm in hand_landmarks.landmark:
                        row.append(lm.x)
                        row.append(lm.y)
                    # Append label
                    row.append(gesture_name)
                    csv_writer.writerow(row)

                    collected_count += 1
                    # Check if we've reached the needed_count
                    if collected_count >= needed_count:
                        print(f"Reached a total of {existing_count + collected_count} for '{gesture_name}'.")
                        break

            cv2.imshow("Data Collection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User pressed 'q'. Exiting data collection.")
                break

            if collected_count >= needed_count:
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished collecting {collected_count} new samples for gesture '{gesture_name}'.\n")

    # Re-analyze dataset to see changes
    analyze_dataset(OUTPUT_CSV)
    print("Done.\n")

if __name__ == "__main__":
    main()
