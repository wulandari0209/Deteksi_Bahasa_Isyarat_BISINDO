// =============================================================
//  KONFIGURASI
// =============================================================
const FLASK_URL = "http://127.0.0.1:5000";
const SEQUENCE_LENGTH = 60; // frame untuk LSTM

const RF_VOTE_WINDOW = 5; // ambil 5 frame, pilih yang terbanyak
const RF_HOLD_FRAMES = 12; // frame stabil sebelum auto-tambah ke kalimat
const RF_COOLDOWN_MS = 1500; // jeda setelah auto-tambah

// =============================================================
//  STATE
// =============================================================
let currentMode = "lstm";
let sequence = []; // buffer LSTM
let rfVoteBuffer = []; // buffer vote RF
let isPredicting = false;
let lastWord = "";
let sentenceWords = [];

// RF stabilizer
let rfHoldCount = 0;
let rfLastLabel = "";
let rfCooldown = false;

// =============================================================
//  DOM ELEMENTS
// =============================================================
const videoEl = document.getElementById("webcam");
const canvasEl = document.getElementById("output_canvas");
const ctx = canvasEl.getContext("2d");
const predEl = document.getElementById("prediction");
const confEl = document.getElementById("confidence");
const confFill = document.getElementById("conf-fill");
const frameCount = document.getElementById("frame-count");
const infoBox = document.getElementById("info-box");
const holdWrap = document.getElementById("hold-bar-wrap");
const holdFill = document.getElementById("hold-bar-fill");
const holdLabel = document.getElementById("hold-bar-label");
const voteRow = document.getElementById("vote-row");
const voteChips = document.getElementById("vote-chips");
const debugPanel = document.getElementById("debug-panel");

canvasEl.width = 640;
canvasEl.height = 480;

// =============================================================
//  MODE SWITCH
// =============================================================
function switchMode(mode) {
  currentMode = mode;
  sequence = [];
  rfVoteBuffer = [];
  rfHoldCount = 0;
  rfLastLabel = "";

  document.getElementById("mode-lstm").className =
    mode === "lstm" ? "mode-btn active" : "mode-btn";
  document.getElementById("mode-rf").className =
    mode === "rf" ? "mode-btn active" : "mode-btn";
  document.getElementById("mode-badge").textContent =
    mode.toUpperCase() + " MODE";
  document.getElementById("label-type").textContent =
    mode === "rf" ? "Huruf / Angka Terdeteksi" : "Kata Terdeteksi";

  infoBox.style.display = mode === "lstm" ? "block" : "none";
  holdWrap.style.display = mode === "rf" ? "block" : "none";
  voteRow.style.display = mode === "rf" ? "flex" : "none";

  holdFill.style.width = "0%";
  predEl.textContent = "—";
}

// =============================================================
//  MEDIAPIPE CALLBACK
// =============================================================
function onResults(results) {
  ctx.save();
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  // Gambar kamera di-mirror untuk tampilan visual
  ctx.translate(canvasEl.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(results.image, 0, 0, canvasEl.width, canvasEl.height);

  if (results.multiHandLandmarks?.length) {
    const allHands = results.multiHandLandmarks;
    const handedness = results.multiHandedness || [];

    // Gambar landmark semua tangan
    for (const lms of allHands) drawLandmarks(lms);

    let rightUserHand = null; // tangan kanan user → slot 0
    let leftUserHand = null; // tangan kiri user  → slot 1

    for (let i = 0; i < allHands.length; i++) {
      const mpLabel = (handedness[i]?.label || "").toLowerCase();
      if (mpLabel === "left") {
        rightUserHand = allHands[i];
      } else if (mpLabel === "right") {
        leftUserHand = allHands[i];
      } else {
        if (!rightUserHand) rightUserHand = allHands[i];
        else if (!leftUserHand) leftUserHand = allHands[i];
      }
    }

    // Ekstrak fitur RF
    let features_rf = [];
    for (const lms of [rightUserHand, leftUserHand]) {
      if (lms) {
        const xs = lms.map((lm) => lm.x);
        const ys = lms.map((lm) => lm.y);
        const min_x = Math.min(...xs);
        const min_y = Math.min(...ys);

        for (const lm of lms) {
          features_rf.push(lm.x - min_x);
          features_rf.push(lm.y - min_y);
        }
      } else {
        for (let k = 0; k < 42; k++) features_rf.push(0);
      }
    }

    if (features_rf.length !== 84) {
      dbg(`[WARN] features_rf.length = ${features_rf.length} (bukan 84!)`);
    }

    // Ekstrak fitur LSTM
    let features_lstm = [];
    const lstmHands = [];
    if (rightUserHand) lstmHands.push(rightUserHand);
    if (leftUserHand) lstmHands.push(leftUserHand);
    const handsForLstm =
      lstmHands.length > 0 ? lstmHands : allHands.slice(0, 2);

    for (const lms of handsForLstm) {
      const wrist = lms[0];
      for (const lm of lms) {
        features_lstm.push(lm.x - wrist.x);
        features_lstm.push(lm.y - wrist.y);
        features_lstm.push(lm.z - wrist.z);
      }
    }
    if (handsForLstm.length < 2) {
      for (let k = 0; k < 63; k++) features_lstm.push(0);
    }

    // Kirim ke backend Flask
    if (currentMode === "lstm") {
      sequence.push(features_lstm);
      if (sequence.length > SEQUENCE_LENGTH) sequence.shift();
      frameCount.textContent = sequence.length;

      if (sequence.length === SEQUENCE_LENGTH && !isPredicting) {
        sendToFlask("/predict", { keypoints: [...sequence] }, handleLstmResult);
      }
    } else {
      if (!isPredicting) {
        sendToFlask("/predict_rf", { features: features_rf }, handleRfResult);
      }
    }
  } else {
    if (currentMode === "lstm") {
      frameCount.textContent = sequence.length;
    }
  }

  ctx.restore();
}

// =============================================================
//  API COMMUNICATION (FETCH FLASK)
// =============================================================
async function sendToFlask(endpoint, payload, callback) {
  isPredicting = true;
  try {
    const res = await fetch(`${FLASK_URL}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    dbg(`${endpoint} → ${JSON.stringify(data)}`);
    if (callback) callback(data);
  } catch (e) {
    dbg(`ERROR ${endpoint}: ${e.message}`);
  } finally {
    setTimeout(
      () => {
        isPredicting = false;
      },
      currentMode === "rf" ? 80 : 0,
    );
  }
}

// =============================================================
//  PREDICTION HANDLERS
// =============================================================
function handleLstmResult(data) {
  if (!data.prediction) return;
  const conf = data.confidence;
  updateResultUI(data.prediction, conf);
  lastWord = data.prediction;

  if (conf > 0.8) addHistory(data.prediction, conf);
}

function handleRfResult(data) {
  if (!data.prediction) return;

  const label = data.prediction;
  const conf = data.confidence;

  rfVoteBuffer.push(label);
  if (rfVoteBuffer.length > RF_VOTE_WINDOW) rfVoteBuffer.shift();

  renderVoteChips(rfVoteBuffer);

  const winner = majority(rfVoteBuffer);

  updateResultUI(winner, conf);
  lastWord = winner;

  // Logika auto-tambah ke kalimat (hold stabil)
  if (winner === rfLastLabel) {
    rfHoldCount++;
  } else {
    rfLastLabel = winner;
    rfHoldCount = 0;
  }

  const pct = Math.min((rfHoldCount / RF_HOLD_FRAMES) * 100, 100);
  holdFill.style.width = pct + "%";
  holdLabel.textContent = `Tahan: ${winner} (${rfHoldCount}/${RF_HOLD_FRAMES})`;

  if (rfHoldCount >= RF_HOLD_FRAMES && !rfCooldown) {
    sentenceWords.push(winner);
    renderSentence();
    addHistory(winner, conf);
    rfHoldCount = 0;
    holdFill.style.width = "0%";
    rfCooldown = true;
    setTimeout(() => {
      rfCooldown = false;
    }, RF_COOLDOWN_MS);
  }
}

// =============================================================
//  UTILITY HELPERS
// =============================================================
function majority(arr) {
  const freq = {};
  let max = 0;
  let winner = arr[arr.length - 1];
  for (const v of arr) {
    freq[v] = (freq[v] || 0) + 1;
    if (freq[v] > max) {
      max = freq[v];
      winner = v;
    }
  }
  return winner;
}

function updateResultUI(label, conf) {
  const pct = Math.round((conf || 0) * 100);
  predEl.textContent = label;
  predEl.style.color = "#000000";
  confEl.textContent = pct + "%";
  confFill.style.width = pct + "%";
}

function renderVoteChips(buf) {
  const freq = {};
  for (const v of buf) freq[v] = (freq[v] || 0) + 1;
  const win = majority(buf);
  voteChips.innerHTML = Object.entries(freq)
    .map(
      ([k, v]) =>
        `<span class="vote-chip${k === win ? " winner" : ""}">${k}×${v}</span>`,
    )
    .join("");
}

function drawLandmarks(lms) {
  const bones = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [0, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [0, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [0, 17],
    [17, 18],
    [18, 19],
    [19, 20],
    [5, 9],
    [9, 13],
    [13, 17],
  ];
  ctx.strokeStyle = "rgba(0,0,0,0.8)";
  ctx.lineWidth = 2;
  for (const [a, b] of bones) {
    ctx.beginPath();
    ctx.moveTo(lms[a].x * canvasEl.width, lms[a].y * canvasEl.height);
    ctx.lineTo(lms[b].x * canvasEl.width, lms[b].y * canvasEl.height);
    ctx.stroke();
  }
  for (const lm of lms) {
    ctx.beginPath();
    ctx.arc(lm.x * canvasEl.width, lm.y * canvasEl.height, 4, 0, 2 * Math.PI);
    ctx.fillStyle = "#000000";
    ctx.fill();
  }
}

// =============================================================
//  SENTENCE MANAGEMENT
// =============================================================
function renderSentence() {
  document.getElementById("sentence-words").innerHTML = sentenceWords
    .map((w) => `<span class="chip">${w}</span>`)
    .join("");
}

function addWord() {
  if (!lastWord) return;
  sentenceWords.push(lastWord);
  renderSentence();
}

function addSpace() {
  sentenceWords.push(" ");
  renderSentence();
}

function clearAll() {
  sentenceWords = [];
  renderSentence();
}

function addHistory(word, conf) {
  const list = document.getElementById("history-list");
  const item = document.createElement("div");
  item.className = "h-item";
  item.innerHTML = `<b>${word}</b> <span>${((conf || 0) * 100).toFixed(1)}%</span>`;
  list.prepend(item);
  if (list.children.length > 8) list.lastChild.remove();
}

// =============================================================
//  DEBUG PANEL LOGS
// =============================================================
function dbg(msg) {
  debugPanel.textContent = msg + "\n" + debugPanel.textContent;
  if (debugPanel.textContent.length > 2000)
    debugPanel.textContent = debugPanel.textContent.slice(0, 2000);
}

function toggleDebug() {
  debugPanel.style.display =
    debugPanel.style.display === "none" ? "block" : "none";
}

// =============================================================
//  MEDIAPIPE INITIALIZATION
// =============================================================
const handsMP = new Hands({
  locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`,
});

handsMP.setOptions({
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
});
handsMP.onResults(onResults);

const camera = new Camera(videoEl, {
  onFrame: async () => {
    await handsMP.send({ image: videoEl });
  },
  width: 640,
  height: 480,
});
camera.start();

// Initialize UI layout state on load
switchMode("lstm");