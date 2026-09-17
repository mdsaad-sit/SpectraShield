import { useState, useRef, useCallback } from 'react';
import { startLiveSession, predictChunk, stopLiveSession } from '../services/api';

const CHUNK_DURATION = 4; // seconds

/**
 * Encode Float32 PCM samples into a WAV Blob.
 */
function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

/**
 * Concatenate an array of Float32Arrays into one contiguous Float32Array.
 */
function concatFloat32Arrays(arrays) {
  let totalLength = 0;
  for (let i = 0; i < arrays.length; i++) {
    totalLength += arrays[i].length;
  }
  const result = new Float32Array(totalLength);
  let offset = 0;
  for (let i = 0; i < arrays.length; i++) {
    result.set(arrays[i], offset);
    offset += arrays[i].length;
  }
  return result;
}

/**
 * useLiveMonitoring — Full live microphone monitoring lifecycle.
 *
 * Manages: getUserMedia → capture 4s chunks → POST each chunk → accumulate results
 * Also records all audio for playback after stopping.
 *
 * IMPORTANT: Uses the ACTUAL AudioContext sample rate (not a hardcoded value)
 * to correctly size chunks and communicate with the backend. The backend will
 * resample to 16 kHz internally if needed.
 *
 * FIX: The ScriptProcessorNode is connected through a zero-gain GainNode
 * instead of directly to audioContext.destination. This prevents the
 * microphone signal from being played back through the speakers (which
 * would create acoustic feedback contaminating the captured signal).
 *
 * FIX: Chunk prediction requests are queued and processed sequentially.
 * Chunk N+1 is not sent until chunk N's response has been received.
 * This ensures deterministic ordering for cumulative voting.
 */
export function useLiveMonitoring() {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [chunkResults, setChunkResults] = useState([]);
  const [latestResult, setLatestResult] = useState(null);
  const [error, setError] = useState(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [audioLevel, setAudioLevel] = useState(0);
  const [recordingUrl, setRecordingUrl] = useState(null);

  const streamRef = useRef(null);
  const contextRef = useRef(null);
  const processorRef = useRef(null);
  // Buffer accumulator: array of Float32Array fragments
  const bufferFragmentsRef = useRef([]);
  const bufferLengthRef = useRef(0);
  const recordedSamplesRef = useRef([]);
  const chunkIndexRef = useRef(0);
  const sessionIdRef = useRef(null);
  const isMonitoringRef = useRef(false);
  const timerRef = useRef(null);
  const actualSampleRateRef = useRef(16000);
  const chunkSamplesRef = useRef(16000 * CHUNK_DURATION);
  // Sequential chunk processing queue
  const chunkQueueRef = useRef([]);
  const isProcessingChunkRef = useRef(false);
  const gainNodeRef = useRef(null);

  const generateSessionId = () => {
    return `live-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  };

  /**
   * Drain the chunk queue one item at a time.
   * Each chunk must complete (success or failure) before the next is sent.
   */
  const drainChunkQueue = useCallback(async () => {
    if (isProcessingChunkRef.current) return;
    if (chunkQueueRef.current.length === 0) return;

    isProcessingChunkRef.current = true;

    while (chunkQueueRef.current.length > 0) {
      const { audioData, chunkIndex, sid, sampleRate } = chunkQueueRef.current.shift();

      try {
        const data = await predictChunk(sid, audioData, sampleRate, chunkIndex);
        setChunkResults((prev) => [...prev, data]);
        setLatestResult(data);
      } catch (err) {
        console.error(`Chunk ${chunkIndex} prediction failed:`, err);
        setError(`Chunk ${chunkIndex + 1} prediction failed: ${err.message || 'Unknown error'}`);
        // Do NOT add a fake result to chunkResults.
        // Do NOT silently swallow the error.
      }
    }

    isProcessingChunkRef.current = false;
  }, []);

  /**
   * Enqueue a chunk for sequential processing.
   * Audio capture is never blocked — only the API requests are serialized.
   */
  const enqueueChunk = useCallback((audioData, chunkIndex, sid, sampleRate) => {
    chunkQueueRef.current.push({ audioData, chunkIndex, sid, sampleRate });
    drainChunkQueue();
  }, [drainChunkQueue]);

  const start = useCallback(async () => {
    setError(null);
    setChunkResults([]);
    setLatestResult(null);
    setElapsedTime(0);
    bufferFragmentsRef.current = [];
    bufferLengthRef.current = 0;
    recordedSamplesRef.current = [];
    chunkIndexRef.current = 0;
    chunkQueueRef.current = [];
    isProcessingChunkRef.current = false;

    // Revoke previous recording URL to free memory
    if (recordingUrl) {
      URL.revokeObjectURL(recordingUrl);
      setRecordingUrl(null);
    }

    try {
      // Request microphone — do NOT constrain sampleRate here;
      // let the browser use its native rate for best quality.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });

      streamRef.current = stream;

      // Start backend session
      const sid = generateSessionId();
      sessionIdRef.current = sid;
      setSessionId(sid);

      await startLiveSession(sid);

      // Set up Web Audio API — use browser's default sample rate.
      // The backend will resample to 16 kHz internally.
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      contextRef.current = audioContext;

      // Read the ACTUAL sample rate the browser is using
      const actualRate = audioContext.sampleRate;
      actualSampleRateRef.current = actualRate;
      const chunkSamples = Math.floor(actualRate * CHUNK_DURATION);
      chunkSamplesRef.current = chunkSamples;

      console.log(`[SpectraShield] AudioContext sampleRate: ${actualRate}, chunk size: ${chunkSamples} samples`);

      const source = audioContext.createMediaStreamSource(stream);

      // Use ScriptProcessorNode for chunk capture
      const bufferSize = 4096;
      const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (!isMonitoringRef.current) return;

        const inputData = e.inputBuffer.getChannelData(0);
        // Copy the input buffer — it is reused by the browser between calls
        const samples = new Float32Array(inputData);

        // Calculate audio level for visualizer
        let sum = 0;
        for (let i = 0; i < samples.length; i++) {
          sum += Math.abs(samples[i]);
        }
        setAudioLevel(sum / samples.length);

        // Save all samples for recording playback
        recordedSamplesRef.current.push(new Float32Array(samples));

        // Accumulate samples using typed array fragments (no spread into plain Array)
        bufferFragmentsRef.current.push(samples);
        bufferLengthRef.current += samples.length;

        // When we have CHUNK_DURATION seconds of audio, process the chunk
        if (bufferLengthRef.current >= chunkSamplesRef.current) {
          const targetLen = chunkSamplesRef.current;

          // Concatenate all fragments into one contiguous Float32Array
          const fullBuffer = concatFloat32Arrays(bufferFragmentsRef.current);

          // Extract exactly targetLen samples for this chunk
          const chunkData = fullBuffer.slice(0, targetLen);

          // Keep any leftover samples for the next chunk
          if (fullBuffer.length > targetLen) {
            bufferFragmentsRef.current = [fullBuffer.slice(targetLen)];
            bufferLengthRef.current = fullBuffer.length - targetLen;
          } else {
            bufferFragmentsRef.current = [];
            bufferLengthRef.current = 0;
          }

          const currentIndex = chunkIndexRef.current;
          chunkIndexRef.current += 1;

          // Enqueue for sequential processing — never overlaps with previous chunk
          enqueueChunk(chunkData, currentIndex, sessionIdRef.current, actualSampleRateRef.current);
        }
      };

      source.connect(processor);

      // FIX: Connect processor through a ZERO-GAIN GainNode to prevent
      // microphone audio from playing back through speakers.
      //
      // ScriptProcessorNode requires being connected to the audio graph
      // destination to keep firing onaudioprocess events. A zero-gain
      // GainNode acts as a silent sink — the processor stays active
      // but no sound reaches the speakers.
      //
      // WITHOUT this fix, the microphone signal is routed to speakers,
      // creating acoustic feedback that contaminates the captured audio
      // with reverb/comb-filtering artifacts. The CNN model interprets
      // these artifacts as synthetic speech characteristics, causing
      // every chunk to be classified as FAKE.
      const silentGain = audioContext.createGain();
      silentGain.gain.value = 0;
      silentGain.connect(audioContext.destination);
      processor.connect(silentGain);
      gainNodeRef.current = silentGain;

      setIsMonitoring(true);
      isMonitoringRef.current = true;

      // Elapsed time timer
      timerRef.current = setInterval(() => {
        setElapsedTime((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      if (err.name === 'NotAllowedError') {
        setError('Microphone permission denied. Please allow microphone access.');
      } else if (err.name === 'NotFoundError') {
        setError('No microphone found. Please connect a microphone.');
      } else {
        setError(err.message || 'Failed to start live monitoring');
      }
    }
  }, [enqueueChunk, recordingUrl]);

  const stop = useCallback(async () => {
    isMonitoringRef.current = false;
    setIsMonitoring(false);
    setAudioLevel(0);

    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }

    if (gainNodeRef.current) {
      gainNodeRef.current.disconnect();
      gainNodeRef.current = null;
    }

    if (contextRef.current) {
      contextRef.current.close();
      contextRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    // Build WAV from recorded samples
    const chunks = recordedSamplesRef.current;
    if (chunks.length > 0) {
      const merged = concatFloat32Arrays(chunks);
      const wavBlob = encodeWAV(merged, actualSampleRateRef.current);
      const url = URL.createObjectURL(wavBlob);
      setRecordingUrl(url);
    }

    if (sessionIdRef.current) {
      try {
        await stopLiveSession(sessionIdRef.current);
      } catch {
        // Ignore stop errors
      }
    }

    bufferFragmentsRef.current = [];
    bufferLengthRef.current = 0;
    chunkQueueRef.current = [];
    isProcessingChunkRef.current = false;
  }, []);

  const reset = useCallback(() => {
    stop();
    setChunkResults([]);
    setLatestResult(null);
    setError(null);
    setElapsedTime(0);
    setSessionId(null);
    recordedSamplesRef.current = [];

    if (recordingUrl) {
      URL.revokeObjectURL(recordingUrl);
      setRecordingUrl(null);
    }
  }, [stop, recordingUrl]);

  return {
    isMonitoring,
    sessionId,
    chunkResults,
    latestResult,
    error,
    elapsedTime,
    audioLevel,
    recordingUrl,
    start,
    stop,
    reset,
  };
}
