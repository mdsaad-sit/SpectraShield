import { useHealth } from '../../hooks/useHealth';
import './HealthBadge.css';

export default function HealthBadge() {
  const { isConnected, modelLoaded, device, error } = useHealth();

  return (
    <div className={`health-badge glass-card-static ${isConnected ? 'health-connected' : 'health-disconnected'}`}>
      <div className="health-badge-header">
        <span className={`status-dot ${isConnected ? 'status-dot-online' : 'status-dot-offline'}`} />
        <span className="health-badge-title">
          {isConnected ? 'Backend Connected' : 'Backend Offline'}
        </span>
      </div>

      {isConnected && (
        <div className="health-badge-details">
          <div className="health-badge-item">
            <span className="health-badge-label">Model</span>
            <span className={`badge ${modelLoaded ? 'badge-real' : 'badge-fake'}`}>
              {modelLoaded ? '✓ Loaded' : '✗ Missing'}
            </span>
          </div>
          {device && (
            <div className="health-badge-item">
              <span className="health-badge-label">Device</span>
              <span className="badge badge-neutral mono">{device.toUpperCase()}</span>
            </div>
          )}
        </div>
      )}

      {error && (
        <p className="health-badge-error">{error}</p>
      )}
    </div>
  );
}
