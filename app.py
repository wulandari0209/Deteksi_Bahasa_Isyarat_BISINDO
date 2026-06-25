from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# =====================================================
# 1. LOAD MODEL & ENCODER LSTM
# =====================================================

MODEL_PATH   = r'D:\huruf_bisindo\bahasa_isyarat_web\hasil_model_lstm.h5'
ENCODER_PATH = r'D:\huruf_bisindo\bahasa_isyarat_web\label_encoder.pickle'

model_lstm = tf.keras.models.load_model(MODEL_PATH)

with open(ENCODER_PATH, 'rb') as f:
    label_encoder_lstm = pickle.load(f)




# =====================================================
# 2. LOAD MODEL RANDOM FOREST
# =====================================================

RF_MODEL_PATH = r'D:\huruf_bisindo\bahasa_isyarat_web\model.p'

with open(RF_MODEL_PATH, 'rb') as f:
    rf_obj = pickle.load(f)

model_rf = rf_obj['model'] if isinstance(rf_obj, dict) else rf_obj



# 3. LABEL MAP — ADAPTASI STRUKTUR FOLDER BARU (HURUF & ANGKA ASLI)
LABEL_MAP = {}
for c in model_rf.classes_:
    class_str = str(c)
    if class_str.isalpha():
        LABEL_MAP[class_str] = class_str.upper()
    else:
        LABEL_MAP[class_str] = class_str




# =====================================================
# ROUTE HOME
# =====================================================

@app.route('/')
def index():
    return render_template('index.html')


# =====================================================
# ENDPOINT LSTM
# =====================================================

@app.route('/predict', methods=['POST'])
def predict_lstm():
    try:
        data      = request.json
        keypoints = data.get('keypoints')

        if not keypoints:
            return jsonify({'error': 'Tidak ada data keypoints'}), 400

        input_array = np.array(keypoints, dtype=np.float32)

        expected_shape = model_lstm.input_shape  # (None, 60, 126)
        if len(expected_shape) == 3:
            expected_seq  = expected_shape[1]
            expected_feat = expected_shape[2]
            if input_array.shape != (expected_seq, expected_feat):

                return jsonify({
                    'error': f'Shape fitur salah: {input_array.shape}, '
                             f'butuh ({expected_seq}, {expected_feat})'
                }), 400

        input_data = np.expand_dims(input_array, axis=0)

        prediction            = model_lstm(input_data, training=False).numpy()
        predicted_class_index = int(np.argmax(prediction, axis=1)[0])
        confidence            = float(prediction[0][predicted_class_index])
        predicted_word        = label_encoder_lstm.inverse_transform([predicted_class_index])[0]


        return jsonify({'prediction': predicted_word, 'confidence': confidence})

    except Exception as e:

        return jsonify({'error': str(e)}), 500


# =====================================================
# ENDPOINT RANDOM FOREST
# =====================================================

@app.route('/predict_rf', methods=['POST'])
def predict_rf():
    try:
        data     = request.json
        features = data.get('features')

        if not features:
            return jsonify({'error': 'Tidak ada data fitur'}), 400

        # SINKRONISASI PADDING URUTAN FITUR (Sesuai dengan 4_test_model.py lokal)
        # Sisi python membutuhkan struktur array tepat 84 kolom fitur
        if len(features) < 84:
            features.extend([0] * (84 - len(features)))
        elif len(features) > 84:
            features = features[:84]

        input_data = np.array([features], dtype=np.float32)

        # Prediksi: model mengembalikan karakter folder asli (misal langsung: 'a', 'b', '1', ...)
        raw_prediction = str(model_rf.predict(input_data)[0])

        # Terjemahkan huruf kecil ke kapital menggunakan LABEL_MAP baru
        predicted_label = LABEL_MAP.get(raw_prediction, raw_prediction)

        # Confidence Score
        try:
            prob       = model_rf.predict_proba(input_data)
            confidence = float(np.max(prob))
        except Exception:
            confidence = 1.0



        return jsonify({'prediction': predicted_label, 'confidence': confidence})

    except Exception as e:

        return jsonify({'error': str(e)}), 500


# =====================================================
# DEBUG ENDPOINT RANDOM FOREST
@app.route('/debug_rf')
def debug_rf():
    return jsonify({
        'num_classes' : len(LABEL_MAP),
        'classes_raw' : [str(c) for c in model_rf.classes_],
        'label_map'   : LABEL_MAP,
        'model_type'  : str(type(model_rf))
    })


# =====================================================
# DEBUG ENDPOINT LSTM
@app.route('/debug_lstm')
def debug_lstm():
    try:
        classes = list(label_encoder_lstm.classes_)
    except Exception:
        classes = []
    return jsonify({
        'input_shape'  : str(model_lstm.input_shape),
        'output_shape' : str(model_lstm.output_shape),
        'num_classes'  : len(classes),
        'classes'      : classes,
    })


# =====================================================
# MAIN
if __name__ == '__main__':
    app.run(debug=True, port=5000)