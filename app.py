import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from collections import deque

# ======================================
# Load Model + Label Encoder
# ======================================

model = tf.keras.models.load_model("gesture_model.keras")

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ======================================
# MediaPipe Setup
# ======================================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ======================================
# Prediction Smoothing
# ======================================

prediction_history = deque(maxlen=10)

# ======================================
# Webcam
# ======================================

cap = cv2.VideoCapture(0)

print("Real-time Gesture Recognition Started. Press 'q' to quit.")

# ======================================
# Main Loop
# ======================================

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    gesture_name = "No Hand Detected"
    confidence = 0.0

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # ======================================
        # Extract 63 Features (x, y, z)
        # ======================================

        wrist = hand_landmarks.landmark[0]

        landmarks = []

        for lm in hand_landmarks.landmark:
            landmarks.extend([
                lm.x - wrist.x,
                lm.y - wrist.y,
                lm.z - wrist.z
            ])

        landmarks = np.array(landmarks, dtype=np.float32)

        # Normalize
        max_val = np.max(np.abs(landmarks))
        if max_val != 0:
            landmarks = landmarks / max_val

        # ======================================
        # Prediction
        # ======================================

        input_data = np.expand_dims(landmarks, axis=0)

        prediction = model.predict(input_data, verbose=0)[0]

        pred_index = np.argmax(prediction)
        confidence = prediction[pred_index]

        prediction_history.append(pred_index)

        # Smooth prediction (majority vote)
        final_index = max(set(prediction_history), key=prediction_history.count)

        if confidence > 0.85:
            gesture_name = label_encoder.classes_[final_index]
        else:
            gesture_name = "Unknown"

    # ======================================
    # UI Overlay
    # ======================================

    cv2.rectangle(frame, (0, 0), (400, 120), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Gesture: {gesture_name}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ======================================
# Cleanup
# ======================================

cap.release()
cv2.destroyAllWindows()