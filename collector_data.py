import cv2
import mediapipe as mp
import csv
import os
import numpy as np

# ==========================
# Configuration
# ==========================
GESTURE_NAME = "Thumbs_Up"      # Change for each gesture
DATA_FILE = "gesture_data.csv"
TOTAL_SAMPLES = 500             # Number of samples to collect
SAVE_INTERVAL = 3               # Save every 3 frames

# ==========================
# Initialize MediaPipe
# ==========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ==========================
# Create dataset if needed
# ==========================
if not os.path.exists(DATA_FILE):
    print("Creating new dataset...")
else:
    print("Appending to existing dataset...")

cap = cv2.VideoCapture(0)

sample_count = 0
frame_count = 0

print(f"\nCollecting data for: {GESTURE_NAME}")
print(f"Target Samples : {TOTAL_SAMPLES}")
print("Press 'q' to Quit\n")

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        wrist = hand_landmarks.landmark[0]

        landmarks = []

        for lm in hand_landmarks.landmark:

            landmarks.extend([
                lm.x - wrist.x,
                lm.y - wrist.y,
                lm.z - wrist.z
            ])

        landmarks = np.array(landmarks, dtype=np.float32)

        max_value = np.max(np.abs(landmarks))

        if max_value != 0:
            landmarks = landmarks / max_value

        frame_count += 1

        if frame_count % SAVE_INTERVAL == 0:

            with open(DATA_FILE, "a", newline="") as f:

                writer = csv.writer(f)

                writer.writerow([GESTURE_NAME] + landmarks.tolist())

            sample_count += 1

    # ==========================
    # Display Information
    # ==========================

    cv2.rectangle(frame, (0, 0), (420, 120), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Gesture : {GESTURE_NAME}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Samples : {sample_count}/{TOTAL_SAMPLES}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to Exit",
        (10, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cv2.imshow("Gesture Data Collector", frame)

    if sample_count >= TOTAL_SAMPLES:
        print(f"\nCollected {TOTAL_SAMPLES} samples for {GESTURE_NAME}")
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nDataset collection completed.")