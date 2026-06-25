import os
import cv2

# ========================
# CONFIG
# ========================
DATA_DIR = './data'
labels = ['halo', 'maaf', 'permisi', 'tolong', 'cantik']  # ganti sesuai kebutuhan
VIDEOS_PER_CLASS = 50
SEQUENCE_LENGTH = 60  # jumlah frame per video
FPS = 20

# ========================
# SETUP
# ========================
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

cap = cv2.VideoCapture(0)

# ambil ukuran frame kamera
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# ========================
# LOOP PER KELAS
# ========================
for label in labels:
    class_path = os.path.join(DATA_DIR, label)
    os.makedirs(class_path, exist_ok=True)

    print(f'\n=== Kelas: {label} ===')

    for vid_num in range(VIDEOS_PER_CLASS):
        print(f'Merekam video {vid_num+1}/{VIDEOS_PER_CLASS}')

        # ========================
        # WAIT SEBELUM RECORD
        # ========================
        while True:
            ret, frame = cap.read()
            cv2.putText(frame, f'{label} - Tekan Q untuk mulai', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Preview', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # ========================
        # SET VIDEO WRITER
        # ========================
        video_path = os.path.join(class_path, f'{vid_num}.mp4')
        out = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            FPS,
            (frame_width, frame_height)
        )

        # ========================
        # RECORD FRAME
        # ========================
        frame_count = 0

        while frame_count < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.putText(frame, f'Recording {label} [{frame_count+1}/{SEQUENCE_LENGTH}]',
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Recording', frame)

            out.write(frame)
            frame_count += 1

            if cv2.waitKey(25) & 0xFF == 27:  # ESC untuk keluar
                break

        out.release()

cap.release()
cv2.destroyAllWindows()

print("Pengambilan dataset selesai!")