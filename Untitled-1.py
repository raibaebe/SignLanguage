import cv2
import mediapipe as mp
import numpy as np
import os
import json

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks_from_video(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    all_landmarks = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip and convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            # Take first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract landmark coordinates normalized (x,y,z)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            
            all_landmarks.append(landmarks)
            
            # Optional: draw landmarks on frame for visualization
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save landmarks as numpy array or json
    np.save(save_path, np.array(all_landmarks))
    print(f"Saved landmarks to {save_path}")

# Usage example (replace with your own video file)
extract_landmarks_from_video("asl_video.mp4", "asl_landmarks.npy")





# Simulated example: load multiple landmark sequences and labels

X = []  # sequences of landmarks
y = []  # labels (integers or strings for sign classes)

for sign_class, video_file in [("hello", "hello.mp4"), ("thank_you", "thank_you.mp4")]:
    save_file = f"{sign_class}_landmarks.npy"
    extract_landmarks_from_video(video_file, save_file)
    landmarks_seq = np.load(save_file)
    X.append(landmarks_seq)
    y.append(sign_class)

# Now you have X as list of sequences and y as labels





from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Flatten landmarks per frame
X_flat = [landmarks.reshape(len(landmarks), -1) for landmarks in X]

# Pad sequences to max length
max_len = max(len(seq) for seq in X_flat)
X_padded = pad_sequences(X_flat, maxlen=max_len, dtype='float32', padding='post', truncating='post')

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking

num_classes = len(label_encoder.classes_)
input_shape = (max_len, 63)  # sequence length x features

model = Sequential([
    Masking(mask_value=0., input_shape=input_shape),
    LSTM(64, return_sequences=False),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()


# Train the model
model.fit(X_padded, y_encoded, epochs=20, batch_size=8, validation_split=0.2)


def predict_sign_from_live(model, label_encoder):
    cap = cv2.VideoCapture(0)
    sequence = []
    max_len = 30  # or your trained sequence length
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            sequence.append(landmarks)
            
            if len(sequence) > max_len:
                sequence.pop(0)
            
            # Prepare input for model
            input_seq = np.array(sequence)
            input_seq = np.expand_dims(input_seq, axis=0)
            input_seq = pad_sequences(input_seq, maxlen=max_len, dtype='float32', padding='post', truncating='post')
            
            prediction = model.predict(input_seq)
            predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
            
            cv2.putText(frame, f"Sign: {predicted_class[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        cv2.imshow("Sign Language Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Usage
predict_sign_from_live(model, label_encoder)
 