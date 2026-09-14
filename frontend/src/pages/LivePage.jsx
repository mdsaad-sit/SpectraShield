import { useLiveMonitoring } from '../hooks/useLiveMonitoring';
import AudioVisualizer from '../components/live/AudioVisualizer';
import ChunkTimeline from '../components/live/ChunkTimeline';
import LiveResult from '../components/live/LiveResult';
import './LivePage.css';

export default function LivePage() {
  const {
    isMonitoring,
    chunkResults,
    latestResult,
    error,
    elapsedTime,
    audioLevel,
    recordingUrl,
    start,
    stop,
    reset,
  } = useLiveMonitoring();

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="page live-page">
      <div className="page-header">
        <h1 className="page-title">Live Monitoring</h1>
        <p className="page-subtitle">
          Monitor audio in real-time through your microphone
        </p>
      </div>

      {/* Control Section */}
      <div className="live-control-section" style={{ animation: 'fadeInUp 0.4s ease-out' }}>
        <div className="live-control glass-card-static">
          {/* Main Control Button */}
          <div className="live-control-main">
            <button
              className={`live-toggle-btn ${isMonitoring ? 'live-toggle-active' : ''}`}
              onClick={isMonitoring ? stop : start}
            >
              <div className="live-toggle-icon">
                {isMonitoring ? (
                  <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                    <rect x="8" y="8" width="16" height="16" rx="2" fill="currentColor" />
                  </svg>
                ) : (
                  <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                    <circle cx="16" cy="12" r="4" stroke="currentColor" strokeWidth="2" />
                    <path d="M8 16a8 8 0 0016 0" stroke="currentColor" strokeWidth="2" strokeLinecap="round" fill="none" />
                    <path d="M16 20v6M12 26h8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                  </svg>
                )}
              </div>
              {isMonitoring ? (
                <div className="live-toggle-pulse" />
              ) : null}
            </button>

            <div className="live-control-info">
              <h3 className="live-control-status">
                {isMonitoring ? 'Monitoring Active' : 'Start Monitoring'}
              </h3>
              <p className="live-control-hint">
                {isMonitoring
                  ? `Recording — ${formatTime(elapsedTime)}`
                  : 'Click to start capturing audio from your microphone'}
              </p>
            </div>

            {isMonitoring && (
              <button className="btn btn-danger" onClick={stop}>
                Stop
              </button>
            )}
          </div>

          {/* Audio Visualizer */}
          <AudioVisualizer audioLevel={audioLevel} isActive={isMonitoring} />
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="live-error glass-card-static" style={{ animation: 'fadeInUp 0.3s ease-out' }}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="10" stroke="#ef4444" strokeWidth="1.5" />
            <path d="M12 8v4M12 16h.01" stroke="#ef4444" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
          <div>
            <h3 style={{ color: 'var(--color-fake)', fontSize: 'var(--font-size-base)', fontWeight: 600, marginBottom: 'var(--space-1)' }}>
              Error
            </h3>
            <p style={{ color: 'var(--color-text-secondary)', fontSize: 'var(--font-size-sm)' }}>
              {error}
            </p>
          </div>
          <button className="btn btn-outline" onClick={reset} style={{ marginLeft: 'auto' }}>Dismiss</button>
        </div>
      )}

      {/* Live Result — show during monitoring AND after stopping */}
      {latestResult && (
        <div style={{ animation: 'fadeInUp 0.4s ease-out' }}>
          <LiveResult latestResult={latestResult} elapsedTime={elapsedTime} />
        </div>
      )}

      {/* Chunk Timeline */}
      {chunkResults.length > 0 && (
        <div className="live-timeline-section glass-card-static" style={{ animation: 'fadeInUp 0.4s ease-out 0.1s both' }}>
          <ChunkTimeline chunkResults={chunkResults} />
        </div>
      )}

      {/* Instructions (when idle) */}
      {!isMonitoring && chunkResults.length === 0 && !error && (
        <div className="live-instructions" style={{ animation: 'fadeInUp 0.5s ease-out 0.3s both' }}>
          <div className="live-instructions-grid">
            <div className="live-instruction-card glass-card-static">
              <div className="live-instruction-step">1</div>
              <h4>Start Monitoring</h4>
              <p>Click the microphone button above to begin capturing audio</p>
            </div>
            <div className="live-instruction-card glass-card-static">
              <div className="live-instruction-step">2</div>
              <h4>Wait 20 Seconds</h4>
              <p>The system collects 5 chunks of 4 seconds each before producing a result</p>
            </div>
            <div className="live-instruction-card glass-card-static">
              <div className="live-instruction-step">3</div>
              <h4>View Results</h4>
              <p>Real-time REAL/FAKE classification updates after every 4-second chunk</p>
            </div>
          </div>

          <div className="live-specs glass-card-static">
            <h4 className="live-specs-title">Live Monitoring Specifications</h4>
            <div className="live-specs-grid">
              <div className="live-spec">
                <span className="live-spec-value mono">4s</span>
                <span className="live-spec-label">Chunk Duration</span>
              </div>
              <div className="live-spec">
                <span className="live-spec-value mono">5</span>
                <span className="live-spec-label">Min. Chunks</span>
              </div>
              <div className="live-spec">
                <span className="live-spec-value mono">20s</span>
                <span className="live-spec-label">First Result</span>
              </div>
              <div className="live-spec">
                <span className="live-spec-value mono">16kHz</span>
                <span className="live-spec-label">Sample Rate</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Audio Playback — shown after stopping */}
      {!isMonitoring && recordingUrl && (
        <div className="live-playback glass-card-static" style={{ animation: 'fadeInUp 0.4s ease-out 0.2s both' }}>
          <div className="live-playback-header">
            <div className="live-playback-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="#60a5fa" strokeWidth="1.5" />
                <path d="M10 8l6 4-6 4V8z" fill="#60a5fa" />
              </svg>
            </div>
            <div>
              <h3 className="live-playback-title">Session Recording</h3>
              <p className="live-playback-subtitle">Play back the monitored audio</p>
            </div>
            <a
              href={recordingUrl}
              download={`spectrashield-live-${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.wav`}
              className="btn btn-outline"
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M8 2v8M4 7l4 4 4-4M3 12v1.5a.5.5 0 00.5.5h9a.5.5 0 00.5-.5V12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Download
            </a>
          </div>
          <audio
            controls
            src={recordingUrl}
            className="live-playback-audio"
          />
        </div>
      )}

      {/* Reset button when stopped but has results */}
      {!isMonitoring && chunkResults.length > 0 && (
        <div className="live-reset-section" style={{ animation: 'fadeInUp 0.3s ease-out' }}>
          <button className="btn btn-primary btn-lg" onClick={reset}>
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path d="M4 10a6 6 0 1011.65-2M16 4v4h-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            New Session
          </button>
        </div>
      )}
    </div>
  );
}
