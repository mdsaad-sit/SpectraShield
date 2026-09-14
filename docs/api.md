# SpectraShield API Documentation

## Base URL
- Development: `http://localhost:5000`
- Frontend: `http://localhost:5173`

## Environment Variables
- `VITE_API_BASE_URL=http://localhost:5000`
- `VITE_SOCKET_URL=http://localhost:5000`
- `FLASK_ENV=development`
- `FRONTEND_URL=http://localhost:5173`
- `MODEL_PATH=backend/model/best_model.pth`
- `MODEL_CONFIG_PATH=backend/model/model_config.json`

## HTTP Status Codes
- `200` - Success
- `400` - Bad request (missing/invalid audio)
- `500` - Internal error (model not loaded, inference error)

---

## Endpoints

### GET /api/health

**Purpose:** Check backend health and model status.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda"
}
```

**Without model:**
```json
{
  "status": "ok",
  "model_loaded": false,
  "device": "cpu"
}
```

**Errors:** None - endpoint always returns 200.

---

### POST /api/predict

**Purpose:** Classify recorded audio (REAL vs FAKE).

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Fields: `audio` (file)

**Successful Response:**
```json
{
  "mode": "recorded",
  "final_prediction": "FAKE",
  "confidence": 0.96,
  "total_chunks": 5,
  "real_chunks": 1,
  "fake_chunks": 4,
  "chunks": [
    {
      "chunk_index": 1,
      "start_time": 0,
      "end_time": 5,
      "prediction": "FAKE",
      "fake_probability": 0.94,
      "confidence": 0.94
    }
  ]
}
```

**Error Responses:**
- No audio provided: `400` `{"error": "No audio file provided"}`
- Model not loaded: `500` `{"error": "Trained model files are not installed. Place best_model.pth and model_config.json inside backend/model/"}`
- Invalid/corrupted audio: `400` `{"error": "Cannot decode audio file"}`

---

### WebSocket Events (Live Monitoring)

#### Connection Events

**connect**
- Emitted when client connects via Socket.IO
- No parameters required

**disconnect**
- Emitted when client disconnects
- No parameters required

#### Audio Events

**audio_chunk**
- Emitted by frontend when a 4-second audio chunk is ready
- Payload: `{ session_id: string, chunk_index: number, audio_data: float32_array, sample_rate: number }`

**Response: window_result**
- Emitted by backend after processing each chunk
- Payload: `{ window_number: number, chunk_index: number, prediction: "FAKE"\|"REAL", fake_probability: number, confidence: number, elapsed_time: number }`

#### Session Events

**session_start**
- Emitted when live monitoring begins
- Payload: `{ session_id: string }`

**session_reset**
- Emitted when monitoring stops and resets
- Payload: `{ session_id: string }`

---

## Chunk Rules

### Recorded Audio
- Audio split into exactly 5-second chunks at 16 kHz
- 80,000 samples per chunk
- Final partial chunk MUST be padded (not discarded)
- Each chunk independently classified
- Majority voting determines final result

### Live Monitoring
- Audio divided into 4-second chunks at 16 kHz
- 64,000 samples per chunk
- **First 20 seconds (5 chunks):** no final result displayed
- **After 20 seconds:** majority voting on latest 5-chunk window
- **Rolling window:** chunks 1-5 → 2-6 → 3-7 → etc.
- Result updated after every 4-second chunk after initial 20-second wait

## Threshold

- **CLASSIFICATION_THRESHOLD = 0.30** (locked, do not change)
- `fake_probability >= 0.30` → FAKE
- `fake_probability < 0.30` → REAL
- Do not use 0.5 as threshold

## Voting Logic

### Majority Voting Rules:
1. Count REAL chunks and FAKE chunks
2. If FAKE count > REAL count → final prediction = FAKE
3. If REAL count > FAKE count → final prediction = REAL
4. If equal counts (tie): use mean fake probability
   - If mean_fake_prob >= 0.30 → FAKE
   - Otherwise → REAL

### Confidence:
- `confidence = (matching_chunks / total_chunks)`
- e.g., 4 FAKE / 5 total → confidence = 0.8

## Error Handling

### Backend Errors:
- Missing audio file: 400 error with message
- Unsupported file format: 400 error
- Empty audio: 400 error
- Corrupted audio: 400 error
- Model not loaded: 500 error with clear message
- Inference error: 500 error, no stack traces exposed to user
- Preprocessing error: 500 error

### Frontend Errors:
- Microphone permission denied: show error message
- Microphone unavailable: show error message
- Backend unavailable: show offline/error state
- Model missing: show health warning
- Socket disconnect: handle gracefully, show reconnection info
- Upload failure: show error message
- Invalid audio: show error message
- Processing failure: show error message

## Model Files

- `best_model.pth`: Trained model weights (binary, not committed to Git)
- `model_config.json`: Model configuration (not committed to Git)
- Each developer must place these files in `backend/model/` manually
- Application starts without them; health endpoint reports `model_loaded: false`