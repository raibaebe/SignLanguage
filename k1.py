import os
import glob
import numpy as np
import pyttsx3
import cv2
import mediapipe as mp

from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import (
    Input, Masking, Bidirectional, LSTM, Dropout,
    Dense, Activation, Multiply, Lambda
)
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K

# ----------------------------------------
# 1) LOAD PRECOMPUTED LANDMARK SEQUENCES
# ----------------------------------------
BASE_PREP = "data/WLASL100/preprocessing"
SPLITS = {
    "train": os.path.join(BASE_PREP, "train", "pose"),
    "val":   os.path.join(BASE_PREP, "val",   "pose"),
    "test":  os.path.join(BASE_PREP, "test",  "pose"),
}
for split, path in SPLITS.items():
    if not os.path.isdir(path):
        raise RuntimeError(f"Missing folder '{split}': {path}")

def load_split(path):
    X, y = [], []
    for gloss in os.listdir(path):
        gdir = os.path.join(path, gloss)
        if not os.path.isdir(gdir): continue
        for npy_file in glob.glob(os.path.join(gdir, "*.npy")):
            seq = np.load(npy_file)  # shape (T,126)
            if seq.size:
                X.append(seq)
                y.append(gloss)
    return X, y

X_tr, y_tr = load_split(SPLITS["train"])
X_vl, y_vl = load_split(SPLITS["val"])
X_te, y_te = load_split(SPLITS["test"])
print(f"Loaded sequences → Train: {len(X_tr)}, Val: {len(X_vl)}, Test: {len(X_te)}")
if not X_tr:
    raise RuntimeError("No training sequences found.")

# ----------------------------------------
# 2) DETERMINE MAX LENGTH AND PAD
# ----------------------------------------
max_len = max(len(s) for s in X_tr)
print("Max sequence length:", max_len)

def pad_list(seqs):
    return pad_sequences(seqs, maxlen=max_len, dtype="float32", padding="post")

X_trp = pad_list(X_tr)
X_vlp = pad_list(X_vl)
X_tep = pad_list(X_te)

# ----------------------------------------
# 3) AUGMENT WITH VELOCITY & ACCELERATION
# ----------------------------------------
def augment_with_dynamics(X):
    augmented = []
    for s in X:
        zero = np.zeros((1,126), dtype="float32")
        s0   = np.vstack([zero, s[:-1]])
        vel  = s - s0
        acc  = np.vstack([zero, vel[1:] - vel[:-1]])
        augmented.append(np.concatenate([s, vel, acc], axis=-1))  # (max_len, 378)
    return np.stack(augmented)

X_tr_aug = augment_with_dynamics(X_trp)
X_vl_aug = augment_with_dynamics(X_vlp)
X_te_aug = augment_with_dynamics(X_tep)

# ----------------------------------------
# 4) ENCODE LABELS
# ----------------------------------------
le     = LabelEncoder()
y_trn  = le.fit_transform(y_tr)
y_vln  = le.transform(y_vl)
y_tst  = le.transform(y_te)
n_cls  = len(le.classes_)
print("Classes:", le.classes_)

# ----------------------------------------
# 5) BUILD Bi-LSTM + ATTENTION MODEL
# ----------------------------------------
inp   = Input(shape=(max_len, 378))
mask  = Masking(0.0)(inp)
lstm  = Bidirectional(LSTM(64, return_sequences=True))(mask)
lstm  = Dropout(0.3)(lstm)

# Luong-style attention: one score per timestep
score = Dense(1)(lstm)                         # (batch, max_len, 1)
weights = Activation('softmax', name='att_weights')(score)  # softmax over time axis
# context = sum(weight * H) along time
context = Multiply()([lstm, weights])          # (batch, max_len, 128)
context = Lambda(lambda x: K.sum(x, axis=1), name='context')(context)  # (batch,128)

dense1 = Dense(64, activation='relu')(context)
drop1  = Dropout(0.3)(dense1)
out    = Dense(n_cls, activation='softmax')(drop1)

model = Model(inputs=inp, outputs=out)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
model.summary()

# ----------------------------------------
# 6) TRAIN & EVALUATE
# ----------------------------------------
model.fit(
    X_tr_aug, y_trn,
    validation_data=(X_vl_aug, y_vln),
    epochs=30,
    batch_size=8,
    verbose=1
)
loss, acc = model.evaluate(X_te_aug, y_tst, verbose=0)
print(f"Test accuracy: {acc:.4f}")

# ----------------------------------------
# 7) LIVE DEMO WITH DYNAMICS
# ----------------------------------------
tts = pyttsx3.init(); tts.setProperty("rate",150)
mp_hands   = mp.solutions.hands
hands_proc = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2,
                           min_detection_confidence=0.5)
mp_draw    = mp.solutions.drawing_utils
# … after training & evaluation …

# 7) LIVE DEMO WITH DYNAMICS (fixed dims)
tts = pyttsx3.init(); tts.setProperty("rate",150)
mp_hands   = mp.solutions.hands
hands_proc = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)
mp_draw    = mp.solutions.drawing_utils

def live_demo():
    cap, seq_pos, last = cv2.VideoCapture(0), [], None
    # infer expected dynamic feature size (should be 3×pos_dim)
    dyn_dim = model.input_shape[-1]
    pos_dim = dyn_dim // 3

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        f   = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        res = hands_proc.process(rgb)

        if res.multi_hand_landmarks:
            # extract raw landmarks (pos vector)
            data = []
            for i in range(2):
                if i < len(res.multi_hand_landmarks):
                    lm = res.multi_hand_landmarks[i]
                    coords = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
                else:
                    coords = np.zeros(21 * 3, dtype="float32")
                data.append(coords)
            vec = np.concatenate(data)  # length pos_dim

            seq_pos.append(vec)
            if len(seq_pos) > max_len:
                seq_pos.pop(0)

            if len(seq_pos) == max_len:
                # build dynamics
                s = np.stack(seq_pos)      # (max_len, pos_dim)
                zero = np.zeros((1, pos_dim), dtype="float32")
                s0   = np.vstack([zero, s[:-1]])
                vel  = s - s0
                # compute acceleration
                acc = np.vstack([zero, vel[1:] - vel[:-1]])
                # concatenate [pos, vel, acc]
                dyn = np.concatenate([s, vel, acc], axis=1)  # (max_len, dyn_dim)
                inp = dyn[np.newaxis, ...]                   # (1, max_len, dyn_dim)

                pred = model.predict(inp, verbose=0)[0]
                idx  = np.argmax(pred)
                conf = pred[idx]

                if conf > 0.5 and idx != last:
                    cls  = le.inverse_transform([idx])[0]
                    tts.say(cls.replace("_", " "))
                    tts.runAndWait()
                    last = idx

                if last is not None:
                    label = le.inverse_transform([last])[0]
                    cv2.putText(
                        f, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2
                    )

            for lm in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(f, lm, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Live Sign Recognition", f)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

live_demo()

