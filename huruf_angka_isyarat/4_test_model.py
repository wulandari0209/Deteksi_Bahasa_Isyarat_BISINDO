import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
labels_dict = {i: labels[i] for i in range(len(labels))}

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.3
)

while True:
    data_aux = []

    ret, frame = cap.read()
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hands_landmarks = results.multi_hand_landmarks

        while len(hands_landmarks) < 2:
            hands_landmarks.append(None)

        x_all = []
        y_all = []

        for hand_landmarks in hands_landmarks:
            if hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

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

                x_all.extend(x_)
                y_all.extend(y_)

            else:
                data_aux.extend([0] * 42)

        x1 = int(min(x_all) * W) - 10
        y1 = int(min(y_all) * H) - 10
        x2 = int(max(x_all) * W) - 10
        y2 = int(max(y_all) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = str(prediction[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
