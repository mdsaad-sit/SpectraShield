import './LiveResult.css';

export default function LiveResult({ latestResult, elapsedTime }) {
  if (!latestResult) return null;

  const { final_result_available, final_prediction, final_confidence, real_chunks, fake_chunks, total_chunks, mean_fake_probability, message } = latestResult;

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Before 20 seconds — collecting phase
  if (!final_result_available) {
    const chunksCollected = total_chunks || 0;
    const chunksNeeded = 5;
    const progress = (chunksCollected / chunksNeeded) * 100;

    return (
      <div className="live-result glass-card-static live-result-collecting">
        <div className="live-result-collecting-inner">
          {/* Progress Ring */}
          <div className="progress-ring-container">
            <svg className="progress-ring" width="120" height="120" viewBox="0 0 120 120">
              <circle
                className="progress-ring-bg"
                cx="60" cy="60" r="52"
                fill="none" strokeWidth="6"
              />
              <circle
                className="progress-ring-fill"
                cx="60" cy="60" r="52"
                fill="none" strokeWidth="6"
                strokeDasharray={`${2 * Math.PI * 52}`}
                strokeDashoffset={`${2 * Math.PI * 52 * (1 - progress / 100)}`}
                strokeLinecap="round"
              />
            </svg>
            <div className="progress-ring-text">
              <span className="progress-ring-value mono">{chunksCollected}/{chunksNeeded}</span>
              <span className="progress-ring-label">chunks</span>
            </div>
          </div>

          <div className="live-result-collecting-info">
            <h3 className="live-result-collecting-title">Collecting Audio…</h3>
            <p className="live-result-collecting-subtitle">
              {message || `Final classification will be available after 20 seconds.`}
            </p>
            <div className="live-result-collecting-meta">
              <span className="mono">{formatTime(elapsedTime)}</span>
              <span className="live-result-collecting-sep">•</span>
              <span>{chunksCollected * 4}s of 20s minimum</span>
            </div>
          </div>
        </div>

        {/* Current chunk prediction */}
        {latestResult.prediction && (
          <div className="live-result-current">
            <span className="section-label">Latest Chunk</span>
            <div className="live-result-current-row">
              <span className={`badge ${latestResult.prediction === 'FAKE' ? 'badge-fake' : 'badge-real'}`}>
                {latestResult.prediction}
              </span>
              <span className="mono" style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)' }}>
                {(latestResult.fake_probability * 100).toFixed(1)}% fake
              </span>
            </div>
          </div>
        )}
      </div>
    );
  }

  // After 20 seconds — showing final result
  const isFake = final_prediction === 'FAKE';
  const confidencePercent = (final_confidence * 100).toFixed(1);

  return (
    <div className={`live-result glass-card-static ${isFake ? 'live-result-fake' : 'live-result-real'}`}>
      {/* Verdict */}
      <div className="live-result-verdict">
        <div className={`live-result-verdict-badge ${isFake ? 'verdict-badge-fake' : 'verdict-badge-real'}`}>
          {isFake ? (
            <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
              <circle cx="18" cy="18" r="16" fill="rgba(239,68,68,0.2)" stroke="#ef4444" strokeWidth="1.5" />
              <path d="M12 12l12 12M24 12L12 24" stroke="#ef4444" strokeWidth="2" strokeLinecap="round" />
            </svg>
          ) : (
            <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
              <circle cx="18" cy="18" r="16" fill="rgba(16,185,129,0.2)" stroke="#10b981" strokeWidth="1.5" />
              <path d="M10 18l5 5 11-11" stroke="#10b981" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
        </div>
        <div>
          <h2 className={`live-result-label ${isFake ? 'text-fake' : 'text-real'}`}>
            {final_prediction}
          </h2>
          <p className="live-result-sublabel">
            {isFake ? 'Synthetic Voice Detected' : 'Authentic Voice'}
          </p>
        </div>
        <div className="live-result-confidence">
          <span className={`live-result-confidence-value mono ${isFake ? 'text-fake' : 'text-real'}`}>
            {confidencePercent}%
          </span>
          <span className="live-result-confidence-label">confidence</span>
        </div>
      </div>

      {/* Stats row */}
      <div className="live-result-stats">
        <div className="live-result-stat">
          <span className="live-result-stat-value mono">{formatTime(elapsedTime)}</span>
          <span className="live-result-stat-label">Elapsed</span>
        </div>
        <div className="live-result-stat">
          <span className="live-result-stat-value mono">{total_chunks}</span>
          <span className="live-result-stat-label">Chunks</span>
        </div>
        <div className="live-result-stat">
          <span className="live-result-stat-value mono text-real">{real_chunks}</span>
          <span className="live-result-stat-label">Real</span>
        </div>
        <div className="live-result-stat">
          <span className="live-result-stat-value mono text-fake">{fake_chunks}</span>
          <span className="live-result-stat-label">Fake</span>
        </div>
        <div className="live-result-stat">
          <span className="live-result-stat-value mono">{mean_fake_probability != null ? (mean_fake_probability * 100).toFixed(1) + '%' : '—'}</span>
          <span className="live-result-stat-label">Mean Fake</span>
        </div>
      </div>
    </div>
  );
}
