import './ChunkTimeline.css';

export default function ChunkTimeline({ chunkResults }) {
  if (!chunkResults || chunkResults.length === 0) return null;

  return (
    <div className="chunk-timeline">
      <h3 className="section-label">Chunk History</h3>
      <div className="chunk-timeline-track">
        {chunkResults.map((result, i) => {
          const prediction = result.prediction;
          const isFake = prediction === 'FAKE';
          const prob = result.fake_probability;

          return (
            <div
              key={i}
              className={`chunk-dot ${isFake ? 'chunk-dot-fake' : 'chunk-dot-real'}`}
              title={`Chunk ${i + 1}: ${prediction} (${(prob * 100).toFixed(1)}%)`}
              style={{ animationDelay: `${i * 60}ms` }}
            >
              <span className="chunk-dot-index">{i + 1}</span>
            </div>
          );
        })}
      </div>

      {/* Scrollable detail list */}
      <div className="chunk-detail-list">
        {chunkResults.slice().reverse().map((result, i) => {
          const idx = chunkResults.length - i;
          const isFake = result.prediction === 'FAKE';
          return (
            <div
              key={i}
              className={`chunk-detail-item ${isFake ? 'chunk-detail-fake' : 'chunk-detail-real'}`}
              style={{ animationDelay: `${i * 40}ms` }}
            >
              <span className="chunk-detail-index mono">#{idx}</span>
              <span className={`badge ${isFake ? 'badge-fake' : 'badge-real'}`}>
                {result.prediction}
              </span>
              <span className="chunk-detail-prob mono">
                {(result.fake_probability * 100).toFixed(1)}%
              </span>
              <span className="chunk-detail-time mono">
                {result.elapsed_time}s
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
