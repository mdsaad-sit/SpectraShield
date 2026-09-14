import { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useRecordedAnalysis } from '../hooks/useRecordedAnalysis';
import FileDropZone from '../components/recorded/FileDropZone';
import ResultCard from '../components/recorded/ResultCard';
import './RecordedPage.css';

export default function RecordedPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { upload, result, error, isLoading, reset } = useRecordedAnalysis(
    location.state?.result || null
  );
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileSelect = (file) => {
    setSelectedFile(file);
  };

  const handleAnalyze = () => {
    if (selectedFile) {
      upload(selectedFile);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    reset();
  };

  return (
    <div className="page recorded-page">
      <div className="page-header">
        <h1 className="page-title">Recorded Audio Analysis</h1>
        <p className="page-subtitle">
          Upload an audio file to detect synthetic or AI-generated voice
        </p>
      </div>

      {/* Upload Section */}
      {!result && (
        <div className="recorded-upload-section" style={{ animation: 'fadeInUp 0.4s ease-out' }}>
          <FileDropZone onFileSelect={handleFileSelect} disabled={isLoading} />

          {selectedFile && !isLoading && (
            <div className="recorded-actions" style={{ animation: 'fadeInUp 0.3s ease-out' }}>
              <button className="btn btn-primary btn-lg" onClick={handleAnalyze}>
                <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                  <path
                    d="M16 10H4M4 10l5-5M4 10l5 5"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    transform="rotate(180 10 10)"
                  />
                </svg>
                Analyze Audio
              </button>
              <button className="btn btn-outline" onClick={handleReset}>
                Clear
              </button>
            </div>
          )}

          {/* Loading State */}
          {isLoading && (
            <div className="recorded-loading" style={{ animation: 'fadeIn 0.3s ease-out' }}>
              <div className="recorded-loading-inner glass-card-static">
                <div className="recorded-loading-animation">
                  <div className="recorded-loading-bars">
                    {Array.from({ length: 5 }).map((_, i) => (
                      <div
                        key={i}
                        className="recorded-loading-bar"
                        style={{ animationDelay: `${i * 150}ms` }}
                      />
                    ))}
                  </div>
                </div>
                <div className="recorded-loading-text">
                  <h3>Analyzing Audio…</h3>
                  <p>Processing chunks through SpectraShieldCNN</p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="recorded-error glass-card-static" style={{ animation: 'fadeInUp 0.3s ease-out' }}>
          <div className="recorded-error-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="#ef4444" strokeWidth="1.5" />
              <path d="M12 8v4M12 16h.01" stroke="#ef4444" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </div>
          <div>
            <h3 className="recorded-error-title">Analysis Failed</h3>
            <p className="recorded-error-msg">{error}</p>
          </div>
          <button className="btn btn-outline" onClick={handleReset}>Try Again</button>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="recorded-result-section">
          <ResultCard result={result} />
          <div className="recorded-result-actions" style={{ animation: 'fadeInUp 0.4s ease-out 0.3s both' }}>
            <button className="btn btn-primary btn-lg" onClick={handleReset}>
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                <path d="M4 10a6 6 0 1011.65-2M16 4v4h-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Analyze Another
            </button>
            <button
              className="btn btn-outline btn-lg"
              onClick={() => navigate('/recorded/xai', { state: { result } })}
            >
              XAI
            </button>
          </div>
        </div>
      )}

      {/* Info Section */}
      {!result && !isLoading && (
        <div className="recorded-info" style={{ animation: 'fadeInUp 0.5s ease-out 0.3s both' }}>
          <div className="recorded-info-grid">
            <div className="recorded-info-card glass-card-static">
              <h4 className="recorded-info-title">How it works</h4>
              <ol className="recorded-info-list">
                <li>Upload your audio file</li>
                <li>Audio is split into 5-second chunks</li>
                <li>Each chunk is analyzed by our CNN model</li>
                <li>Majority voting determines the final result</li>
              </ol>
            </div>
            <div className="recorded-info-card glass-card-static">
              <h4 className="recorded-info-title">Specifications</h4>
              <ul className="recorded-info-specs">
                <li><span className="spec-label">Sample Rate</span><span className="mono">16 kHz</span></li>
                <li><span className="spec-label">Chunk Size</span><span className="mono">5 seconds</span></li>
                <li><span className="spec-label">Spectrogram</span><span className="mono">128×128 Log-Mel</span></li>
                <li><span className="spec-label">Threshold</span><span className="mono">0.30</span></li>
                <li><span className="spec-label">Max File Size</span><span className="mono">100 MB</span></li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
