
import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
from tqdm import tqdm

# ========================
# KONFIGURASI
# ========================
VIDEO_DIR      = './data'
SEQUENCE_LENGTH = 60        # harus sama dengan training & inference
SLIDING_STEP   = 15         # sliding window step (semakin kecil → lebih banyak sampel)
FEATURE_SIZE   = 126        # 2 tangan × 21 landmark × (x, y, z)

valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')

# ========================
# MEDIAPIPE
# ========================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

# ========================
# EKSTRAKSI KEYPOINTS (x, y, z — wrist-relative)
# ========================
def extract_keypoints(results):

    data_aux = []

    if results.multi_hand_landmarks:
        detected = list(results.multi_hand_landmarks)

        # Pastikan selalu ada 2 slot tangan
        while len(detected) < 2:
            detected.append(None)

        detected = detected[:2]   # batasi maksimal 2

        for hand_landmarks in detected:
            if hand_landmarks:
                lms = hand_landmarks.landmark

                # Wrist sebagai titik referensi
                wx, wy, wz = lms[0].x, lms[0].y, lms[0].z

                for lm in lms:
                    data_aux.append(lm.x - wx)
                    data_aux.append(lm.y - wy)
                    data_aux.append(lm.z - wz)
            else:
                data_aux.extend([0.0] * 63)   # 21 landmark × 3
    else:
        data_aux.extend([0.0] * 126)

    return data_aux


# ========================
# AUGMENTASI DATA
# ========================
def augment_sequence(seq):
    augmented = []

    # 1. Flip horizontal: balikkan tangan kiri↔kanan
    #    x menjadi -x (karena wrist-relative, flip cukup negasikan x tiap landmark)
    flipped = seq.copy()
    for i in range(0, FEATURE_SIZE, 3):   # setiap komponen x
        flipped[:, i] = -flipped[:, i]
    augmented.append(flipped)

    # 2. Gaussian noise ringan
    noisy = seq + np.random.normal(0, 0.005, seq.shape)
    augmented.append(noisy)

    # 3. Time warp: perlambat 80% → crop
    n = len(seq)
    indices = np.linspace(0, n - 1, int(n * 1.25)).astype(int)
    indices = np.clip(indices, 0, n - 1)
    stretched = seq[indices][:n]   # crop ke panjang asli
    augmented.append(stretched)

    # 4. Time warp: percepat 125% → repeat frame terakhir untuk padding
    indices2 = np.linspace(0, n - 1, int(n * 0.8)).astype(int)
    indices2 = np.clip(indices2, 0, n - 1)
    compressed = seq[indices2]
    pad = np.tile(compressed[-1], (n - len(compressed), 1))
    compressed = np.vstack([compressed, pad])
    augmented.append(compressed)

    return augmented


# ========================
# SLIDING WINDOW
# ========================
def sliding_window(sequence, length, step):
    windows = []
    for start in range(0, len(sequence) - length + 1, step):
        windows.append(sequence[start:start + length])
    return windows


# ========================
# PROSES SEMUA VIDEO
# ========================
data   = []
labels = []

class_list = sorted([
    d for d in os.listdir(VIDEO_DIR)
    if os.path.isdir(os.path.join(VIDEO_DIR, d))
])

print(f"Kelas ditemukan: {class_list}\n")

for label in class_list:
    class_path = os.path.join(VIDEO_DIR, label)
    video_files = [
        f for f in os.listdir(class_path)
        if f.lower().endswith(valid_extensions)
    ]

    print(f"[{label}] — {len(video_files)} video")

    for video_file in tqdm(video_files, desc=f"  {label}", unit="video"):
        video_path = os.path.join(class_path, video_file)
        cap = cv2.VideoCapture(video_path)

        raw_sequence = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize kecil untuk mempercepat mediapipe
            small = cv2.resize(frame, (320, 240))
            frame_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            kp = extract_keypoints(results)
            if len(kp) == FEATURE_SIZE:
                raw_sequence.append(kp)

        cap.release()

        if len(raw_sequence) < SEQUENCE_LENGTH:
            # Padding dengan frame kosong jika terlalu pendek
            while len(raw_sequence) < SEQUENCE_LENGTH:
                raw_sequence.append([0.0] * FEATURE_SIZE)
            raw_sequence = [raw_sequence[:SEQUENCE_LENGTH]]
        else:
            # Sliding window → banyak sampel dari satu video
            raw_sequence = sliding_window(raw_sequence, SEQUENCE_LENGTH, SLIDING_STEP)

        for seq in raw_sequence:
            seq_arr = np.array(seq, dtype=np.float32)

            # Simpan sequence asli
            data.append(seq_arr)
            labels.append(label)

            # Simpan augmentasi
            for aug_seq in augment_sequence(seq_arr):
                data.append(aug_seq.astype(np.float32))
                labels.append(label)

    print(f"  → Total sampel sejauh ini: {len(data)}")

hands.close()

data   = np.array(data,   dtype=np.float32)
labels = np.array(labels)

print(f"\nShape akhir data  : {data.shape}")
print(f"Distribusi kelas  :")
unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {u}: {c} sampel")

with open('data_lstm.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("\nEkstraksi selesai! data_lstm.pickle siap.")
print(f"FEATURE_SIZE per frame = {FEATURE_SIZE}  (pastikan train & inference sama!)")
