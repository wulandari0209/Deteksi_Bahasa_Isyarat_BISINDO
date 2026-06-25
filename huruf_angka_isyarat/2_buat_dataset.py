import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.3
)

DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        all_x = []
        all_y = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        
        if results.multi_hand_landmarks:
            hands_landmarks = list(results.multi_hand_landmarks)
            while len(hands_landmarks) < 2:
                hands_landmarks.append(None)

            for hand_landmarks in hands_landmarks:
                if hand_landmarks is not None:
                    x_ = []
                    y_ = []

                    for lm in hand_landmarks.landmark:
                        x_.append(lm.x)
                        y_.append(lm.y)

                    min_x = min(x_)
                    min_y = min(y_)

                    for lm in hand_landmarks.landmark:
                        data_aux.append(lm.x - min_x)
                        data_aux.append(lm.y - min_y)

                else:
                    # padding tangan kosong (42 fitur nol)
                    data_aux.extend([0] * 42)

            data.append(data_aux)
            labels.append(dir_)

print("Jumlah sampel:", len(data))
print("Jumlah label :", len(labels))
print("Panjang fitur pertama:", len(data[0]) if len(data) > 0 else 0)
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': np.array(data), 'labels': np.array(labels)}, f)