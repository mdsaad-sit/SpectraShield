import { Link } from 'react-router-dom';
import HealthBadge from '../components/common/HealthBadge';
import './HomePage.css';

export default function HomePage() {
  return (
    <div className="page home-page">
      {/* Hero */}
      <section className="hero">
        <div className="hero-glow" />
        <div className="hero-shield">
          <svg width="96" height="96" viewBox="0 0 96 96" fill="none">
            <defs>
              <linearGradient id="hero-shield-grad" x1="0" y1="0" x2="96" y2="96">
                <stop offset="0%" stopColor="#3b82f6" />
                <stop offset="50%" stopColor="#8b5cf6" />
                <stop offset="100%" stopColor="#06b6d4" />
              </linearGradient>
            </defs>
            <path
              d="M48 6L12 24v24c0 23.2 15.36 44.88 36 50.2 20.64-5.32 36-27 36-50.2V24L48 6z"
              fill="url(#hero-shield-grad)"
              opacity="0.12"
              stroke="url(#hero-shield-grad)"
              strokeWidth="2"
            />
            <path
              d="M48 16L20 30v18c0 18.4 11.84 35.52 28 40 16.16-4.48 28-21.6 28-40V30L48 16z"
              fill="none"
              stroke="url(#hero-shield-grad)"
              strokeWidth="2"
            />
            {/* Waveform */}
            <path
              d="M30 48h6l3-10 4 20 3-14 3 8 3-6 3 4h6"
              fill="none"
              stroke="#60a5fa"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>

        <h1 className="hero-title">
          <span className="hero-title-spectra">Spectra</span>
          <span className="hero-title-shield">Shield</span>
        </h1>

        <p className="hero-subtitle">
          Spectrogram-Driven Deep Learning Framework for Synthetic Voice Detection
        </p>

        <p className="hero-description">
          Detect AI-generated and deepfake speech in real-time using advanced CNN analysis
          of Log-Mel spectrograms. Upload a recording or monitor live audio.
        </p>

        <div className="hero-actions">
          <Link to="/recorded" className="btn btn-primary btn-lg">
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path d="M4 16l4-4m0 0l4-4m-4 4V4m0 8h8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" transform="rotate(90 10 10)" />
            </svg>
            Analyze Recording
          </Link>
          <Link to="/live" className="btn btn-outline btn-lg">
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <circle cx="10" cy="7" r="3" stroke="currentColor" strokeWidth="1.5" />
              <path d="M5 10a5 5 0 0010 0" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" fill="none" />
              <path d="M10 13v4M7 17h6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
            Live Monitoring
          </Link>
        </div>
      </section>

      {/* Features */}
      <section className="features">
        <div className="features-grid">
          <div className="feature-card glass-card">
            <div className="feature-icon feature-icon-blue">
              <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
                <rect x="3" y="7" width="22" height="14" rx="3" stroke="currentColor" strokeWidth="1.5" />
                <path d="M9 12v4M12 10v6M15 11v5M18 9v8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
            </div>
            <h3 className="feature-title">Spectrogram Analysis</h3>
            <p className="feature-desc">
              128-band Log-Mel spectrograms capture the full frequency signature of voice audio for deep analysis.
            </p>
          </div>

          <div className="feature-card glass-card">
            <div className="feature-icon feature-icon-purple">
              <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
                <circle cx="14" cy="14" r="11" stroke="currentColor" strokeWidth="1.5" />
                <path d="M14 8v6l4 2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
            </div>
            <h3 className="feature-title">Real-Time Detection</h3>
            <p className="feature-desc">
              Live microphone monitoring with 4-second chunks and cumulative majority voting after 20 seconds.
            </p>
          </div>

          <div className="feature-card glass-card">
            <div className="feature-icon feature-icon-cyan">
              <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
                <path d="M14 3l3 6 7 1-5 5 1 7-6-3-6 3 1-7-5-5 7-1 3-6z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
              </svg>
            </div>
            <h3 className="feature-title">99.67% Accuracy</h3>
            <p className="feature-desc">
              SpectraShieldCNN achieves near-perfect validation accuracy with an optimized 0.30 threshold.
            </p>
          </div>

          <div className="feature-card glass-card">
            <div className="feature-icon feature-icon-green">
              <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
                <path d="M14 3L4 9v8c0 8 4.8 15.4 10 17.5C20.2 32.4 25 25 25 17V9L14 3z" stroke="currentColor" strokeWidth="1.5" />
                <path d="M10 14l3 3 6-6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            <h3 className="feature-title">Majority Voting</h3>
            <p className="feature-desc">
              Robust classification through chunk-level predictions with tie-breaking by mean probability.
            </p>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="how-it-works">
        <h2 className="section-heading">How It Works</h2>
        <div className="pipeline">
          {[
            { step: '01', label: 'Audio Input', desc: 'Upload recording or capture live audio' },
            { step: '02', label: 'Chunking', desc: '5s chunks (recorded) or 4s chunks (live)' },
            { step: '03', label: 'Spectrogram', desc: '128-band Log-Mel spectrogram extraction' },
            { step: '04', label: 'CNN Inference', desc: 'SpectraShieldCNN classifies each chunk' },
            { step: '05', label: 'Voting', desc: 'Majority vote across all chunks' },
            { step: '06', label: 'Result', desc: 'REAL or FAKE with confidence score' },
          ].map(({ step, label, desc }, i) => (
            <div key={step} className="pipeline-step" style={{ animationDelay: `${i * 100}ms` }}>
              <div className="pipeline-step-number mono">{step}</div>
              <div className="pipeline-step-content">
                <h4 className="pipeline-step-label">{label}</h4>
                <p className="pipeline-step-desc">{desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Health Badge */}
      <section className="home-health">
        <h2 className="section-heading">System Status</h2>
        <HealthBadge />
      </section>
    </div>
  );
}
