import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Function to extract hand landmarks from video
def extract_landmarks_from_video(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    all_landmarks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hands_data = []
            for i in range(2):  # Always use 2 hands (pad with zeros if needed)
                if i < len(results.multi_hand_landmarks):
                    hand_landmarks = results.multi_hand_landmarks[i]
                    hand = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                else:
                    hand = np.zeros(21 * 3)
                hands_data.append(hand)

            landmarks = np.concatenate(hands_data)  # 126 features total
            all_landmarks.append(landmarks)

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(all_landmarks) == 0:
        print(f"Warning: No landmarks detected in {video_path}")
    else:
        np.save(save_path, np.array(all_landmarks))
        print(f"Saved landmarks to {save_path}")

# Dataset preparation
X = []
y = []

dataset = [
    ("hello", "hello.mp4"),
    ("thank_you", "thank_you.mp4"),
    ("have_a_nice_day", "have_a_nice_day.mp4")
]

for sign_class, video_file in dataset:
    save_file = f"{sign_class}_landmarks.npy"
    extract_landmarks_from_video(video_file, save_file)

    landmarks_seq = np.load(save_file, allow_pickle=True)

    if landmarks_seq.size == 0:
        print(f"No landmarks detected in {video_file}, skipping...")
        continue

    X.append(landmarks_seq)
    y.append(sign_class)

# Flatten and pad sequences
X_flat = [seq.reshape(len(seq), -1) for seq in X]
max_len = max(len(seq) for seq in X_flat)
X_padded = pad_sequences(
    X_flat,
    maxlen=max_len,
    dtype='float32',
    padding='post',
    truncating='post'
)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Build model
input_shape = (max_len, 126)  # 2 hands × 21 landmarks × 3 coords
num_classes = len(label_encoder.classes_)

model = Sequential([
    Masking(mask_value=0., input_shape=input_shape),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# Train the model
model.fit(
    X_padded,
    y_encoded,
    epochs=20,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)

# Real-time prediction
def predict_sign_from_live(model, label_encoder):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    sequence = []
    max_len = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hands_data = []
            for i in range(2):
                if i < len(results.multi_hand_landmarks):
                    hand_landmarks = results.multi_hand_landmarks[i]
                    hand = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                else:
                    hand = np.zeros(21 * 3)
                hands_data.append(hand)

            landmarks = np.concatenate(hands_data)
            sequence.append(landmarks)

            if len(sequence) > max_len:
                sequence.pop(0)

            input_seq = np.array(sequence)
            input_seq = np.expand_dims(input_seq, axis=0)
            input_seq = pad_sequences(
                input_seq,
                maxlen=max_len,
                dtype='float32',
                padding='post',
                truncating='post'
            )

            prediction = model.predict(input_seq, verbose=0)
            predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])

            cv2.putText(
                frame,
                f"Sign: {predicted_class[0]}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Sign Language Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run live recognition
predict_sign_from_live(model, label_encoder)
