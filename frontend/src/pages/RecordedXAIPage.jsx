import { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './RecordedXAIPage.css';

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function ChunkXAI({ chunk }) {
  const [isOpen, setIsOpen] = useState(false);
  const hasImages = Boolean(
    chunk.original_logmel_image && chunk.gradcam_image
  );

  return (
    <article className={`xai-chunk-card ${isOpen ? 'xai-chunk-card-open' : ''}`}>
      <button
        type="button"
        className="xai-chunk-toggle"
        onClick={() => setIsOpen((open) => !open)}
        aria-expanded={isOpen}
      >
        <span className="xai-chunk-toggle-copy">
          <span className="xai-chunk-title">Chunk {chunk.chunk_index + 1}</span>
          <span className="xai-chunk-time">
            {chunk.start_time}s - {chunk.end_time}s
          </span>
          <span className="xai-chunk-summary">
            <span className={`badge ${chunk.prediction === 'FAKE' ? 'badge-fake' : 'badge-real'}`}>
              {chunk.prediction}
            </span>
            <span>Fake Prob. {formatPercent(chunk.fake_probability)}</span>
            <span>Confidence {formatPercent(chunk.confidence)}</span>
          </span>
        </span>
        <span className="xai-chunk-chevron" aria-hidden="true">
          {isOpen ? '▲' : '▼'}
        </span>
      </button>

      {isOpen && (
        <div className="xai-chunk-content">
          <div className="xai-chunk-details">
            <span>Time: <strong>{chunk.start_time}s - {chunk.end_time}s</strong></span>
            <span>Prediction: <strong>{chunk.prediction}</strong></span>
            <span>Fake Probability: <strong>{formatPercent(chunk.fake_probability)}</strong></span>
            <span>Confidence: <strong>{formatPercent(chunk.confidence)}</strong></span>
          </div>

          <div className="xai-explanation-heading">XAI Explanation</div>
          {hasImages ? (
            <>
              <div className="xai-image-stack">
                <figure className="xai-image-figure">
                  <figcaption>Original Log-Mel Spectrogram</figcaption>
                  <img
                    src={chunk.original_logmel_image}
                    alt={`Original Log-Mel spectrogram for chunk ${chunk.chunk_index + 1}`}
                  />
                </figure>
                <figure className="xai-image-figure">
                  <figcaption>Grad-CAM Overlay</figcaption>
                  <img
                    src={chunk.gradcam_image}
                    alt={`Grad-CAM overlay for chunk ${chunk.chunk_index + 1}`}
                  />
                </figure>
              </div>
              <div className="xai-influence-legend" aria-hidden="true">
                <span>Low Model Influence</span>
                <span className="xai-influence-gradient" />
                <span>High Model Influence</span>
              </div>
              <p className="xai-interpretation">
                <strong>Model Interpretation</strong>
                Highlighted regions represent time-frequency regions that
                contributed strongly to the model&apos;s FAKE logit.
              </p>
            </>
          ) : (
            <p className="xai-error">
              XAI is unavailable for this chunk
              {chunk.gradcam_error ? `: ${chunk.gradcam_error}` : '.'}
            </p>
          )}
        </div>
      )}
    </article>
  );
}

export default function RecordedXAIPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const result = location.state?.result;

  if (!result) {
    return (
      <main className="page recorded-xai-page">
        <div className="xai-empty-state glass-card-static">
          <h1 className="page-title">XAI Analysis</h1>
          <p>No recorded prediction is available to explain.</p>
          <button className="btn btn-primary" onClick={() => navigate('/recorded')}>
            Go to Recorded Audio
          </button>
        </div>
      </main>
    );
  }

  const isFake = result.final_prediction === 'FAKE';

  return (
    <main className="page recorded-xai-page">
      <div className="recorded-xai-header">
        <button
          type="button"
          className="btn btn-outline xai-back-button"
          onClick={() => navigate('/recorded', { state: { result } })}
        >
          <span aria-hidden="true">←</span>
          Back to Results
        </button>
        <h1 className="page-title">XAI Analysis</h1>
        <p className="page-subtitle">Model Explainability for Recorded Audio</p>
      </div>

      <section className="xai-summary glass-card-static">
        <div className="xai-summary-verdict">
          <span className="section-label">Overall Prediction</span>
          <strong className={isFake ? 'text-fake' : 'text-real'}>
            {result.final_prediction}
          </strong>
        </div>
        <div className="xai-summary-confidence">
          <span className="section-label">Confidence</span>
          <strong>{formatPercent(result.confidence)}</strong>
        </div>
        <div className="xai-summary-stats">
          <div><strong>{result.total_chunks}</strong><span>Total Chunks</span></div>
          <div><strong className="text-real">{result.real_chunks}</strong><span>Real Chunks</span></div>
          <div><strong className="text-fake">{result.fake_chunks}</strong><span>Fake Chunks</span></div>
          <div><strong>{formatPercent(result.mean_fake_probability)}</strong><span>Mean Fake Probability</span></div>
        </div>
      </section>

      <section className="xai-chunks-section">
        <div className="xai-section-heading">
          <span className="section-label">Chunk Explanations</span>
          <p>Expand a chunk to compare its input spectrogram with the model explanation.</p>
        </div>
        <div className="xai-chunk-list">
          {result.chunks?.map((chunk) => (
            <ChunkXAI key={chunk.chunk_index} chunk={chunk} />
          ))}
        </div>
      </section>
    </main>
  );
}
