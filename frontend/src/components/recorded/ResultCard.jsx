import './ResultCard.css';

export default function ResultCard({ result }) {
  if (!result) return null;

  const {
    final_prediction,
    confidence,
    mean_fake_probability,
    total_chunks,
    real_chunks,
    fake_chunks,
    chunks,
  } = result;

  const isFake = final_prediction === 'FAKE';
  const confidencePercent = (confidence * 100).toFixed(1);
  const meanFakePercent = (mean_fake_probability * 100).toFixed(1);

  return (
    <div className="result-card glass-card-static" style={{ animation: 'fadeInUp 0.5s ease-out' }}>
      {/* Verdict */}
      <div className={`result-verdict ${isFake ? 'result-verdict-fake' : 'result-verdict-real'}`}>
        <div className="result-verdict-icon">
          {isFake ? (
            <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
              <circle cx="24" cy="24" r="22" fill="rgba(239,68,68,0.15)" stroke="#ef4444" strokeWidth="2" />
              <path d="M16 16l16 16M32 16L16 32" stroke="#ef4444" strokeWidth="2.5" strokeLinecap="round" />
            </svg>
          ) : (
            <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
              <circle cx="24" cy="24" r="22" fill="rgba(16,185,129,0.15)" stroke="#10b981" strokeWidth="2" />
              <path d="M14 24l7 7 13-13" stroke="#10b981" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
        </div>
        <div className="result-verdict-text">
          <h2 className={`result-verdict-label ${isFake ? 'text-fake' : 'text-real'}`}>
            {final_prediction}
          </h2>
          <p className="result-verdict-subtitle">
            {isFake ? 'Synthetic / AI-Generated Voice Detected' : 'Authentic Human Voice'}
          </p>
        </div>
      </div>

      {/* Confidence Bar */}
      <div className="result-confidence">
        <div className="result-confidence-header">
          <span className="section-label">Confidence</span>
          <span className={`result-confidence-value mono ${isFake ? 'text-fake' : 'text-real'}`}>
            {confidencePercent}%
          </span>
        </div>
        <div className="progress-bar-track">
          <div
            className="progress-bar-fill"
            style={{
              width: `${confidencePercent}%`,
              background: isFake
                ? 'linear-gradient(90deg, #ef4444, #f87171)'
                : 'linear-gradient(90deg, #10b981, #34d399)',
            }}
          />
        </div>
      </div>

      {/* Stats Grid */}
      <div className="result-stats">
        <div className="result-stat glass-card-static">
          <span className="result-stat-value mono">{total_chunks}</span>
          <span className="result-stat-label">Total Chunks</span>
        </div>
        <div className="result-stat glass-card-static">
          <span className="result-stat-value mono text-real">{real_chunks}</span>
          <span className="result-stat-label">Real Chunks</span>
        </div>
        <div className="result-stat glass-card-static">
          <span className="result-stat-value mono text-fake">{fake_chunks}</span>
          <span className="result-stat-label">Fake Chunks</span>
        </div>
        <div className="result-stat glass-card-static">
          <span className="result-stat-value mono">{meanFakePercent}%</span>
          <span className="result-stat-label">Mean Fake Prob.</span>
        </div>
      </div>

      {/* Chunk Breakdown Table */}
      {chunks && chunks.length > 0 && (
        <div className="result-chunks">
          <h3 className="section-label">Chunk Breakdown</h3>
          <div className="result-chunks-table-wrapper">
            <table className="result-chunks-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Time</th>
                  <th>Prediction</th>
                  <th>Fake Prob.</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {chunks.map((chunk) => (
                  <tr key={chunk.chunk_index} className={chunk.prediction === 'FAKE' ? 'row-fake' : 'row-real'}>
                    <td className="mono">{chunk.chunk_index + 1}</td>
                    <td className="mono">{chunk.start_time}s — {chunk.end_time}s</td>
                    <td>
                      <span className={`badge ${chunk.prediction === 'FAKE' ? 'badge-fake' : 'badge-real'}`}>
                        {chunk.prediction}
                      </span>
                    </td>
                    <td className="mono">{(chunk.fake_probability * 100).toFixed(1)}%</td>
                    <td className="mono">{(chunk.confidence * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
