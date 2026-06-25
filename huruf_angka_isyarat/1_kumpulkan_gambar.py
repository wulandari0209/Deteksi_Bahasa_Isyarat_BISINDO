import os
import cv2
import string

DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)

# Label 0-9 dan a-z
labels = list("0123456789") + list(string.ascii_uppercase)

dataset_size = 400

cap = cv2.VideoCapture(0)

for class_name in labels:

    # Folder sesuai nama label
    class_path = os.path.join(DATA_DIR, class_name)
    os.makedirs(class_path, exist_ok=True)

    print(f'Collecting data for class "{class_name}"')

    # Tunggu tombol Q
    while True:
        ret, frame = cap.read()

        cv2.putText(
            frame,
            f'Ready for "{class_name}" ? Press Q',
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow('frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Ambil gambar
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()

        cv2.putText(
            frame,
            f'{class_name} : {counter}/{dataset_size}',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        cv2.imwrite(
            os.path.join(class_path, f'{counter}.jpg'),
            frame
        )

        counter += 1

cap.release()
cv2.destroyAllWindows()