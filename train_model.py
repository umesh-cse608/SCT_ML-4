import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# Load Dataset
# ==========================================

DATASET_PATH = "gesture_data.csv"

print("Loading dataset...")

data = pd.read_csv(DATASET_PATH, header=None)

# First column contains gesture labels
X = data.iloc[:, 1:].values.astype(np.float32)
y = data.iloc[:, 0].values

print(f"Total Samples : {len(X)}")
print(f"Total Features: {X.shape[1]}")

# Ensure dataset has 63 features (21 landmarks × x,y,z)
if X.shape[1] != 63:
    raise ValueError(
        f"Dataset has {X.shape[1]} features.\n"
        "Expected 63 features.\n"
        "Please collect data using the updated collector_data.py."
    )

# ==========================================
# Encode Labels
# ==========================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

num_classes = len(label_encoder.classes_)

y_onehot = tf.keras.utils.to_categorical(
    y_encoded,
    num_classes=num_classes
)

# Save label encoder
with open("label_encoder.pkl", "wb") as file:
    pickle.dump(label_encoder, file)

print("\nDetected Gesture Classes:")

for i, label in enumerate(label_encoder.classes_):
    print(f"{i} -> {label}")

# ==========================================
# Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_onehot,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

# ==========================================
# Build Neural Network
# ==========================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(63,)),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(32, activation="relu"),

    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# Callbacks
# ==========================================

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_gesture_model.keras",
    monitor="val_accuracy",
    save_best_only=True
)

# ==========================================
# Train
# ==========================================

print("\nTraining Started...\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# ==========================================
# Evaluate
# ==========================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n===========================")
print(f"Test Accuracy : {accuracy*100:.2f}%")
print("===========================")

# ==========================================
# Classification Report
# ==========================================

predictions = model.predict(X_test)

predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

print("\nClassification Report\n")

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=label_encoder.classes_
    )
)

print("\nConfusion Matrix\n")

print(confusion_matrix(
    true_labels,
    predicted_labels
))

# ==========================================
# Save Final Model
# ==========================================

model.save("gesture_model.keras")

print("\nModel Saved Successfully!")

print("Generated Files:")
print("gesture_model.keras")
print("best_gesture_model.keras")
print("label_encoder.pkl")