import pickle
import cv2
import mediapipe as mp
import numpy as np
import threading
from collections import deque

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.models import load_model

# ========================
# LOAD MODEL LSTM
# ========================
model = load_model('model_lstm.h5')
le = pickle.load(open('label_encoder.pickle', 'rb'))

SEQUENCE_LENGTH = 60
FEATURE_SIZE    = 126   # 2 tangan × 21 landmark × (x, y, z)

# Warm-up
dummy = np.zeros((1, SEQUENCE_LENGTH, FEATURE_SIZE))
model.predict(dummy, verbose=0)

# ========================
# MEDIAPIPE
# ========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# ========================
# SHARED STATE
# ========================
sequence      = deque(maxlen=SEQUENCE_LENGTH)
predicted_label = ""
label_lock    = threading.Lock()
is_predicting = False
predict_lock  = threading.Lock()


# ========================
# EKSTRAKSI KEYPOINTS (harus sama persis dengan 2_extract_lstm.py)
# ========================
def extract_keypoints(results):
    data_aux = []

    if results.multi_hand_landmarks:
        detected = list(results.multi_hand_landmarks)
        while len(detected) < 2:
            detected.append(None)
        detected = detected[:2]

        for hand_landmarks in detected:
            if hand_landmarks:
                lms = hand_landmarks.landmark
                wx, wy, wz = lms[0].x, lms[0].y, lms[0].z
                for lm in lms:
                    data_aux.append(lm.x - wx)
                    data_aux.append(lm.y - wy)
                    data_aux.append(lm.z - wz)
            else:
                data_aux.extend([0.0] * 63)
    else:
        data_aux.extend([0.0] * 126)

    return data_aux


# ========================
# PREDIKSI DI THREAD TERPISAH
# ========================
def run_prediction(seq_snapshot):
    global predicted_label, is_predicting

    input_data = np.expand_dims(seq_snapshot, axis=0)
    prediction = model.predict(input_data, verbose=0)
    label = le.inverse_transform([np.argmax(prediction)])[0]
    confidence = float(np.max(prediction))

    with label_lock:
        predicted_label = f"{label} ({confidence*100:.0f}%)"

    with predict_lock:
        is_predicting = False


# ========================
# KAMERA
# ========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # Linux/Mac: cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    small     = cv2.resize(frame, (320, 240))
    frame_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    results   = hands.process(frame_rgb)

    keypoints = extract_keypoints(results)
    if len(keypoints) == FEATURE_SIZE:
        sequence.append(keypoints)

    # Jalankan prediksi non-blocking
    with predict_lock:
        can_predict = not is_predicting

    if can_predict and len(sequence) == SEQUENCE_LENGTH:
        with predict_lock:
            is_predicting = True
        seq_snapshot = np.array(sequence, dtype=np.float32)
        t = threading.Thread(target=run_prediction, args=(seq_snapshot,), daemon=True)
        t.start()

    # Gambar landmark + bounding box
    if results.multi_hand_landmarks:
        x_all, y_all = [], []

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            for lm in hand_landmarks.landmark:
                x_all.append(lm.x)
                y_all.append(lm.y)

        x1 = max(0, int(min(x_all) * W) - 10)
        y1 = max(0, int(min(y_all) * H) - 10)
        x2 = min(W, int(max(x_all) * W) + 10)
        y2 = min(H, int(max(y_all) * H) + 10)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

        with label_lock:
            label_now = predicted_label

        if label_now:
            cv2.putText(frame, label_now, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    # Progress bar buffer
    buf_len = len(sequence)
    bar_w   = int((buf_len / SEQUENCE_LENGTH) * W)
    cv2.rectangle(frame, (0, H - 12), (bar_w, H), (0, 200, 0), -1)
    cv2.putText(frame, f'Buffer: {buf_len}/{SEQUENCE_LENGTH}', (8, H - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imshow('Sign Language Detection', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        sequence.clear()
        with label_lock:
            predicted_label = ""

cap.release()
cv2.destroyAllWindows()
