import os
import glob
import numpy as np
import cv2
import mediapipe as mp

BASE       = "data/WLASL100/preprocessing"
FRAMES_DIR = os.path.join(BASE, "train", "frames"), os.path.join(BASE, "val", "frames"), os.path.join(BASE, "test", "frames")

# Initialize MediaPipe once
mp_hands   = mp.solutions.hands
hands_proc = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

for split in ("train","val","test"):
    in_root  = os.path.join(BASE, split, "frames")
    out_root = os.path.join(BASE, split, "pose")
    os.makedirs(out_root, exist_ok=True)

    for gloss in os.listdir(in_root):
        in_gloss  = os.path.join(in_root, gloss)
        out_gloss = os.path.join(out_root, gloss)
        if not os.path.isdir(in_gloss):
            continue
        os.makedirs(out_gloss, exist_ok=True)

        for clip in os.listdir(in_gloss):
            in_clip  = os.path.join(in_gloss, clip)
            out_file = os.path.join(out_gloss, f"{clip}.npy")
            if os.path.exists(out_file):
                continue  # skip if already done

            seq = []
            jpgs = sorted(glob.glob(os.path.join(in_clip, "*.jpg")))
            for jpg in jpgs:
                img = cv2.imread(jpg)
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = hands_proc.process(rgb)
                if not res.multi_hand_landmarks:
                    continue
                data = []
                for i in range(2):
                    if i < len(res.multi_hand_landmarks):
                        lm = res.multi_hand_landmarks[i]
                        coords = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
                    else:
                        coords = np.zeros(21*3)
                    data.append(coords)
                seq.append(np.concatenate(data))

            if seq:
                np.save(out_file, np.stack(seq))
                print(f"Saved {out_file} (T={len(seq)})")
