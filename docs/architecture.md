# SpectraShield Architecture Documentation

## Overview

SpectraShield is a spectrogram-driven deep learning framework for synthetic voice detection. The application detects whether an audio recording is real human speech or AI-generated/synthetic/deepfake speech.

## Architecture

```
Frontend (React + Vite + Tailwind CSS)
    ↓
REST API / WebSocket (Flask + Flask-SocketIO)
    ↓
Audio Processing (Librosa, NumPy)
    ↓
Chunking (5s recorded, 4s live)
    ↓
Log-Mel Spectrogram Preprocessing
    ↓
SpectraShieldCNN Inference (PyTorch)
    ↓
Sigmoid → 0.30 Threshold
    ↓
Majority Voting
    ↓
Result (REAL/FAKE with confidence)
```

## Model Architecture (SpectraShieldCNN)

The CNN has 4 convolution blocks:

### Block 1:
- Conv2d(1, 32, kernel_size=3, padding=1)
- BatchNorm2d(32)
- ReLU
- MaxPool2d(2)

### Block 2:
- Conv2d(32, 64, kernel_size=3, padding=1)
- BatchNorm2d(64)
- ReLU
- MaxPool2d(2)

### Block 3:
- Conv2d(64, 128, kernel_size=3, padding=1)
- BatchNorm2d(128)
- ReLU
- MaxPool2d(2)

### Block 4:
- Conv2d(128, 256, kernel_size=3, padding=1)
- BatchNorm2d(256)
- ReLU

### Post-feature layers:
- AdaptiveAvgPool2d((1,1))
- Dropout(0.4)
- Linear(256, 1) → single logit

**Input:** `[batch, 1, 128, 128]`  
**Output:** single logit

### Probability conversion:
- `fake_probability = sigmoid(logit)`
- `0 = REAL`, `1 = FAKE`

### Classification threshold: **0.30**
- `fake_probability >= 0.30` → FAKE
- `fake_probability < 0.30` → REAL

## Audio Preprocessing Pipeline

1. Load audio using `librosa.load(path, sr=16000, mono=True)`
2. Reject empty audio (len(y) == 0 → ValueError)
3. Normalize by maximum absolute amplitude: `y = y / max(|y|)`
4. Generate Mel spectrogram:
   - `n_fft=1024, hop_length=256, n_mels=128, power=2.0`
5. Convert to dB: `librosa.power_to_db(mel, ref=np.max)`
6. Min-max normalize: `(logmel - logmel.min()) / (logmel.max() - logmel.min() + 1e-8)`
7. Resize time axis to exactly 128 frames (pad or truncate)
8. Final shape: `(1, 1, 128, 128)` after adding channel and batch dimensions

## Chunking Rules

### Recorded Audio:
- **5-second chunks** at 16 kHz = 80,000 samples per chunk
- Final partial chunk is **padded**, not discarded
- Every chunk independently processed through: audio → preprocessing → model → sigmoid → threshold → prediction
- Majority voting determines final result

### Live Monitoring:
- **4-second chunks** at 16 kHz = 64,000 samples per chunk
- Five 4-second chunks = one 20-second voting window
- **Rolling 20-second window**: After the first 20 seconds, each subsequent 4-second chunk updates the result using the latest 5 chunks (window rolls: chunks 1-5, then 2-6, then 3-7, etc.)
- **No final REAL/FAKE result before 20 seconds** - UI shows "Collecting audio — final classification will be available after 20 seconds."
- After 20 seconds: result updated after every 4-second chunk

## Inference Pipeline

### Recorded Mode:
1. Upload audio file
2. Split into 5-second chunks (final partial chunk padded)
3. Each chunk → preprocessing → SpectraShieldCNN → sigmoid → 0.30 threshold → chunk prediction
4. Majority voting: count REAL vs FAKE chunks
5. Tie-breaking: mean fake probability >= 0.30 → FAKE, otherwise REAL
6. Return: final prediction, confidence, chunk stats, individual chunk results

### Live Monitoring:
1. Browser microphone access via `navigator.mediaDevices.getUserMedia()`
2. Audio divided into 4-second chunks
3. Each chunk sent to Flask backend via Socket.IO
4. Backend processes chunk through same preprocessing and CNN
5. First 20 seconds (5 chunks): collect predictions, NO final result displayed
6. After 20 seconds: majority voting on 5-chunk window, display result
7. Every subsequent 4-second chunk: rolling window update (latest 5 chunks), display new result
8. Session state reset when monitoring stops

## Voting Logic

### Majority Voting:
- Count REAL chunks and FAKE chunks
- If FAKE > REAL → final prediction FAKE
- If REAL > FAKE → final prediction REAL
- If tie (equal counts): use mean fake probability as tie-breaker
  - If mean_fake_prob >= 0.30 → FAKE
  - Otherwise → REAL

### Confidence Calculation:
- If final prediction is FAKE: confidence = fake_chunks / total_chunks
- If final prediction is REAL: confidence = real_chunks / total_chunks

## API Endpoints

### GET /api/health
- Returns: `{"status": "ok", "model_loaded": true/false, "device": "cuda"/"cpu"}`
- Works even when model files are missing

### POST /api/predict
- Multipart/form-data with field "audio"
- Returns JSON with:
  - mode: "recorded"
  - final_prediction: "FAKE"/"REAL"
  - confidence: float
  - total_chunks: int
  - real_chunks: int
  - fake_chunks: int
  - chunks: [...]

### WebSocket Events (Live Monitoring)
- `connect`: Client connected
- `disconnect`: Client disconnected
- `audio_chunk`: Binary/audio data chunk from browser
- `window_result`: {"window_number": int, "prediction": "FAKE"/"REAL", "confidence": float}
- `session_reset`: Reset session state

## Key Differences: Recorded vs Live Mode

| Aspect | Recorded | Live |
|--------|----------|------|
|Chunk duration | 5 seconds | 4 seconds|
|Chunking method | Fixed 5-second split | Rolling 20-second window|
|Final result timing | After all chunks processed | After 20 seconds (5 chunks)|
|Voting type | Single final vote | Rolling 5-chunk window|
|Session handling | One-shot per upload | Continuous, resets on stop|
|UI behavior | Result immediately after analysis | Progress + 20-second wait |

## Technology Stack

- **Frontend:** React, Vite, Tailwind CSS, Socket.IO client
- **Backend:** Flask, Flask-CORS, Flask-SocketIO
- **ML:** PyTorch, SpectraShieldCNN
- **Audio:** Librosa, NumPy, SciPy, SoundFile
- **Environment:** Python virtual environment, .env configuration