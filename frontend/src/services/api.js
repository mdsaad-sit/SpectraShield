// SpectraShield — API Service Layer
// Maps to all 5 backend endpoints

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';

/**
 * GET /api/health
 * Check backend health and model status.
 */
export async function checkHealth() {
  const res = await fetch(`${API_BASE_URL}/api/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

/**
 * POST /api/predict
 * Upload recorded audio for classification.
 * @param {File} file - Audio file to classify
 */
export async function predictRecorded(file) {
  const formData = new FormData();
  formData.append('audio', file);

  const res = await fetch(`${API_BASE_URL}/api/predict`, {
    method: 'POST',
    body: formData,
  });

  const data = await res.json();

  if (!res.ok) {
    throw new Error(data.error || `Prediction failed: ${res.status}`);
  }

  return data;
}

/**
 * POST /api/start
 * Start a new live monitoring session.
 * @param {string} sessionId - Unique session identifier
 */
export async function startLiveSession(sessionId) {
  const res = await fetch(`${API_BASE_URL}/api/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
  });

  const data = await res.json();

  if (!res.ok) {
    throw new Error(data.error || `Failed to start session: ${res.status}`);
  }

  return data;
}

/**
 * POST /api/predict_chunk
 * Send one live audio chunk for prediction.
 * @param {string} sessionId
 * @param {Float32Array|number[]} audioData
 * @param {number} sampleRate
 * @param {number} chunkIndex
 */
export async function predictChunk(sessionId, audioData, sampleRate, chunkIndex) {
  const res = await fetch(`${API_BASE_URL}/api/predict_chunk`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      audio_data: Array.from(audioData),
      sample_rate: sampleRate,
      chunk_index: chunkIndex,
    }),
  });

  const data = await res.json();

  if (!res.ok) {
    throw new Error(data.error || `Chunk prediction failed: ${res.status}`);
  }

  return data;
}

/**
 * POST /api/stop
 * Stop a live monitoring session.
 * @param {string} sessionId
 */
export async function stopLiveSession(sessionId) {
  const res = await fetch(`${API_BASE_URL}/api/stop`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
  });

  const data = await res.json();

  if (!res.ok) {
    throw new Error(data.error || `Failed to stop session: ${res.status}`);
  }

  return data;
}
