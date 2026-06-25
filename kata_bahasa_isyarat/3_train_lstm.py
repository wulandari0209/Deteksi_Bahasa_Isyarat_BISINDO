import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Bidirectional
)
from tensorflow.keras.utils    import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ========================
# REPRODUCIBILITY
# ========================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ========================
# LOAD DATA
# ========================
data_dict = pickle.load(open('data_lstm.pickle', 'rb'))

X = data_dict['data'].astype(np.float32)
y = data_dict['labels']

print(f"Shape X : {X.shape}")
print(f"Kelas   : {np.unique(y)}")

SEQUENCE_LENGTH = X.shape[1]
FEATURE_SIZE    = X.shape[2]
N_CLASSES       = len(np.unique(y))

# ========================
# ENCODE LABEL
# ========================
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat     = to_categorical(y_encoded, num_classes=N_CLASSES)

# ========================
# STRATIFIED SPLIT: 70% train / 15% val / 15% test
# ========================
X_train, X_temp, y_train, y_temp, ye_train, ye_temp = train_test_split(
    X, y_cat, y_encoded,
    test_size=0.30,
    random_state=SEED,
    stratify=y_encoded
)
X_val, X_test, y_val, y_test, ye_val, ye_test = train_test_split(
    X_temp, y_temp, ye_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=ye_temp
)

print(f"\nTrain : {len(X_train)} sampel")
print(f"Val   : {len(X_val)}   sampel")
print(f"Test  : {len(X_test)}  sampel")

# ========================
# ARSITEKTUR MODEL
# ========================
model = Sequential([
    # Layer 1: Bidirectional LSTM → tangkap pola maju & mundur
    Bidirectional(
        LSTM(128, return_sequences=True),
        input_shape=(SEQUENCE_LENGTH, FEATURE_SIZE)
    ),
    BatchNormalization(),
    Dropout(0.3),

    # Layer 2: LSTM biasa
    LSTM(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),

    # Layer 3: LSTM terakhir
    LSTM(32, return_sequences=False),
    BatchNormalization(),
    Dropout(0.2),

    # Fully-connected head
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(N_CLASSES, activation='softmax')
])

model.summary()

# ========================
# COMPILE
# ========================
optimizer = Adam(learning_rate=1e-3)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ========================
# CALLBACKS
# ========================
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,   # otomatis kembalikan bobot terbaik
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,           # kurangi LR jadi 50% bila val_loss stagnan
        patience=7,
        min_lr=1e-6,
        verbose=1
    )
]

# ========================
# TRAINING
# ========================
print("\nMulai training...\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,              # EarlyStopping akan berhenti sendiri
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

# ========================
# EVALUASI DI TEST SET
# ========================
print("\n--- Evaluasi Test Set ---")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {acc * 100:.2f}%")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

print("\nClassification Report:")
print(classification_report(ye_test, y_pred, target_names=le.classes_))

# ========================
# CONFUSION MATRIX
# ========================
cm = confusion_matrix(ye_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix — Test Set')
plt.ylabel('Label Asli')
plt.xlabel('Label Prediksi')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print("Confusion matrix disimpan: confusion_matrix.png")

# ========================
# GRAFIK AKURASI & LOSS
# ========================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history['accuracy'],     label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
ax1.set_title('Akurasi per Epoch')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'],     label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title('Loss per Epoch')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.close()
print("Grafik training disimpan: training_history.png")

# ========================
# SIMPAN MODEL & ENCODER
# ========================
model.save('model_lstm.h5')

with open('label_encoder.pickle', 'wb') as f:
    pickle.dump(le, f)

print("\nModel disimpan  : model_lstm.h5")
print("Encoder disimpan: label_encoder.pickle")
print("\nTraining selesai!")
