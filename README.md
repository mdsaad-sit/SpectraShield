# SpectraShield

SpectraShield is an audio deepfake detection application. It analyzes recorded audio and live microphone input, converts audio into Log-Mel spectrograms, and classifies speech as `REAL` or `FAKE` with a PyTorch convolutional neural network.

The project also includes Grad-CAM explainability for recorded audio. The XAI view shows the original Log-Mel spectrogram and the Grad-CAM overlay for every five-second chunk.

## What The Project Does

SpectraShield supports two workflows:

- **Recorded Audio:** upload an audio file, split it into five-second chunks, classify every chunk, use majority voting, and inspect Grad-CAM explanations.
- **Live Monitoring:** capture microphone audio in the browser, send four-second chunks to the backend, and perform cumulative voting after the first five chunks.

The final classification is based on the model's FAKE probability:

```text
fake_probability = sigmoid(model_logit)

fake_probability >= 0.30  ->  FAKE
fake_probability <  0.30  ->  REAL
```

The threshold and voting logic are part of the current application contract and should not be changed without retraining and re-evaluation.

## Architecture

```text
Browser (React + Vite)
        |
        | REST / JSON and live monitoring requests
        v
Flask backend
        |
        +-- audio loading and chunking
        +-- Log-Mel preprocessing with librosa
        +-- ModelLoader: loads one SpectraShieldCNN checkpoint
        +-- recorded prediction and majority voting
        +-- Grad-CAM image generation for recorded chunks
        +-- live cumulative session voting
```

### Backend

- `backend/run.py` starts Flask-SocketIO on `127.0.0.1:5000`.
- `backend/app/__init__.py` creates the Flask app, configures CORS and Socket.IO, registers routes, and initializes the model.
- `backend/app/routes/recorded.py` handles `POST /api/predict`.
- `backend/app/routes/live.py` handles live session and chunk endpoints.
- `backend/app/routes/health.py` exposes model health information.
- `backend/app/inference/model_loader.py` loads and caches the trained model.
- `backend/app/inference/gradcam.py` explains the FAKE logit using `model.conv4`.

### Frontend

- React 19 with Vite.
- `frontend/src/services/api.js` calls backend endpoints.
- `frontend/src/pages/RecordedPage.jsx` handles recorded upload and results.
- `frontend/src/pages/RecordedXAIPage.jsx` displays the dedicated XAI view.
- `frontend/src/pages/LivePage.jsx` handles microphone monitoring.
- `frontend/src/components/recorded/ResultCard.jsx` displays the normal recorded result.
- `frontend/src/hooks/` contains upload, health, and live-monitoring state logic.

## CNN Model

The trained model is `SpectraShieldCNN` in `backend/app/inference/model.py`. It receives a tensor shaped `[batch, 1, 128, 128]` and returns one scalar logit. That scalar is the FAKE-class logit.

### Layer-by-layer structure

| Stage | Operation | Output channels / size |
|---|---|---|
| Input | Normalized Log-Mel spectrogram | `1 x 128 x 128` |
| Block 1 | `Conv2d(1, 32, 3, padding=1)` + `BatchNorm2d` + ReLU + `MaxPool2d(2)` | `32 x 64 x 64` |
| Block 2 | `Conv2d(32, 64, 3, padding=1)` + `BatchNorm2d` + ReLU + `MaxPool2d(2)` | `64 x 32 x 32` |
| Block 3 | `Conv2d(64, 128, 3, padding=1)` + `BatchNorm2d` + ReLU + `MaxPool2d(2)` | `128 x 16 x 16` |
| Block 4 | `Conv2d(128, 256, 3, padding=1)` + `BatchNorm2d` + ReLU | `256 x 16 x 16` |
| Pooling | `AdaptiveAvgPool2d((1, 1))` | `256 x 1 x 1` |
| Classifier | Flatten + `Dropout(0.4)` + `Linear(256, 1)` | One FAKE logit |

The checkpoint is loaded from `backend/model/best_model.pth`. The optional metadata file is `backend/model/model_config.json`.

### Model loading and inference

`ModelLoader.initialize()` selects CUDA when available and otherwise uses CPU. It loads the checkpoint once, moves the model to the selected device, and places it in evaluation mode.

Normal prediction uses `torch.no_grad()` and returns the logit and its sigmoid probability. Grad-CAM uses the same already-loaded model. It registers hooks on `conv4`, runs a gradient-enabled forward pass, targets `output[:, 0]`, globally averages the gradients, weights the convolutional activations, applies ReLU, and resizes the result to `128 x 128`.

Grad-CAM describes model behavior. Highlighted regions are not proof that an audio recording is fake.

## Audio Preprocessing

The recorded route uses these parameters:

| Parameter | Value |
|---|---:|
| Sample rate | `16000 Hz` |
| Channels | Mono |
| Recorded chunk duration | `5 seconds` |
| Mel bands (`n_mels`) | `128` |
| FFT size (`n_fft`) | `1024` |
| Hop length | `256` |
| Mel power | `2.0` |
| Target frames | `128` |
| Model input | `[1, 1, 128, 128]` |
| Classification threshold | `0.30` |

For every recorded chunk:

1. The waveform is loaded at 16 kHz and converted to mono.
2. The waveform is normalized by its maximum absolute amplitude when nonzero.
3. `librosa.feature.melspectrogram()` creates a 128-band power Mel spectrogram.
4. `librosa.power_to_db(..., ref=np.max)` converts it to Log-Mel values.
5. The values are min-max normalized.
6. The time axis is interpolated to exactly 128 frames.
7. The result becomes a float32 tensor shaped `[1, 1, 128, 128]`.

The exact tensor used for prediction is also used to create the original Log-Mel XAI image. No second visualization-specific preprocessing pipeline is used.

The shared utility at `backend/app/inference/preprocessing.py` is used by preprocessing tests and whole-file processing. The recorded and live routes contain their own chunk-level preprocessing functions because their resizing behavior is tied to those inference workflows.

## Recorded Audio Flow

```text
Upload audio
  -> load at 16 kHz mono
  -> split into five-second chunks
  -> preprocess each chunk
  -> predict each chunk
  -> classify with threshold 0.30
  -> calculate confidence
  -> majority vote across chunks
  -> return final result and per-chunk data
```

`POST /api/predict` returns final statistics and a `chunks` array. Each chunk includes its prediction, probability, confidence, `original_logmel_image`, and `gradcam_image` as in-memory PNG data URLs.

The normal result page provides `Analyze Another` and one global `XAI` action. The XAI action opens `/recorded/xai`, where each chunk can be independently expanded to compare the original Log-Mel spectrogram with its Grad-CAM overlay.

## Live Monitoring Flow

Live monitoring uses a separate workflow and does not use recorded-audio Grad-CAM images:

- Audio is captured in the browser.
- Chunks are four seconds each.
- The first final result is produced after five chunks, or 20 seconds.
- Predictions remain in the session and voting is cumulative, not rolling-window based.
- Later chunks update the cumulative result.

Important live endpoints are:

- `GET /api/live/health`
- `POST /api/start`
- `POST /api/predict_chunk`
- `POST /api/stop`

## Project Structure

```text
Major-Deepfake/
├── backend/
│   ├── app/
│   │   ├── inference/
│   │   ├── routes/
│   │   └── utils/
│   ├── model/
│   │   ├── best_model.pth
│   │   └── model_config.json
│   ├── tests/
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── .env.example
│   └── run.py
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── pages/
│   │   └── services/
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js
├── docs/
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3.10 or newer recommended
- Node.js 18 or newer recommended
- npm
- The trained model files supplied separately:
  - `backend/model/best_model.pth`
  - `backend/model/model_config.json`

## Environment Configuration

Create the backend environment file from the template:

### Windows PowerShell

```powershell
Copy-Item backend\.env.example backend\.env
```

Then open `backend/.env` and update the paths and settings for your machine. At minimum, verify:

```dotenv
FLASK_ENV=development
FLASK_DEBUG=true
HOST=127.0.0.1
PORT=5000
FRONTEND_URL=http://localhost:5173
MODEL_PATH=C:/path/to/Major-Deepfake/backend/model/best_model.pth
MODEL_CONFIG_PATH=C:/path/to/Major-Deepfake/backend/model/model_config.json
MAX_UPLOAD_SIZE_MB=100
CLASSIFICATION_THRESHOLD=0.30
```

Use forward slashes in Windows paths or escape backslashes. Do not commit `backend/.env`; it may contain machine-specific configuration or secrets.

The frontend environment file is `frontend/.env`. Create it from `frontend/.env.example` if needed:

```powershell
Copy-Item frontend\.env.example frontend\.env
```

Its default value is:

```dotenv
VITE_API_BASE_URL=http://localhost:5000
```

## Backend Setup and Start

From the project root in PowerShell:

```powershell
cd backend
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Update `.env`, place the model files in `backend/model/`, and start the server:

```powershell
python run.py
```

The backend runs at `http://localhost:5000`.

To install development and test dependencies as well:

```powershell
python -m pip install -r requirements-dev.txt
```

## Frontend Setup and Start

In a second terminal:

```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```

The frontend runs at `http://localhost:5173`.

Available frontend commands:

```powershell
npm run dev
npm run build
npm run preview
npm run lint
```

## API Endpoints

- `GET /api/health`: backend and model health.
- `POST /api/predict`: recorded audio upload and classification.
- `GET /api/live/health`: live monitoring health.
- `POST /api/start`: start a live session.
- `POST /api/predict_chunk`: classify one live audio chunk.
- `POST /api/stop`: stop a live session.

The frontend development server proxies `/api` requests to `http://localhost:5000` through `frontend/vite.config.js`.

## Testing

Run backend tests from the backend directory after activating the virtual environment:

```powershell
cd backend
pytest
```

Useful focused checks include:

```powershell
pytest tests/test_model.py
pytest tests/test_preprocessing.py
pytest tests/test_chunking.py
pytest tests/test_voting.py
pytest tests/test_api.py
```

For a manual end-to-end check:

1. Start the backend and confirm `GET /api/health` reports the model as loaded.
2. Start the frontend at port 5173.
3. Upload a recorded audio file.
4. Confirm the final prediction, confidence, chunk table, and majority vote.
5. Open `XAI` and expand multiple chunks independently.
6. Confirm every expanded chunk shows its original Log-Mel image and Grad-CAM overlay.
7. Test microphone permissions and live monitoring separately.

## Troubleshooting

### Model is not loaded

Verify that both model files exist and that `MODEL_PATH` in `backend/.env` points to `best_model.pth`. The API health response includes model status and device information.

### Frontend cannot reach the backend

Confirm that Flask is running on port 5000, Vite is running on port 5173, and `FRONTEND_URL` matches the frontend origin.

### Upload fails

Check that the audio file is not empty, is supported by librosa/soundfile, and does not exceed `MAX_UPLOAD_SIZE_MB`.

### Microphone monitoring fails

Allow microphone access in the browser and use a secure context or localhost. Chrome and Edge are recommended for development.

## Important Constraints

The following are intentionally fixed by the current project design:

- Existing model weights and architecture.
- Recorded preprocessing parameters.
- Four-second live chunks and cumulative live voting.
- Five-second recorded chunks.
- FAKE threshold `0.30`.
- Existing majority voting and confidence calculations.
- The recorded and live API contracts.

Do not retrain or replace the model without updating the evaluation and deployment process.
