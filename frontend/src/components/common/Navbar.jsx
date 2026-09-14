import { NavLink } from 'react-router-dom';
import { useHealth } from '../../hooks/useHealth';
import './Navbar.css';

export default function Navbar() {
  const { isConnected, modelLoaded } = useHealth();

  const getStatusClass = () => {
    if (!isConnected) return 'status-dot-offline';
    if (!modelLoaded) return 'status-dot-warning';
    return 'status-dot-online';
  };

  const getStatusText = () => {
    if (!isConnected) return 'Offline';
    if (!modelLoaded) return 'Model Missing';
    return 'Online';
  };

  return (
    <nav className="navbar">
      <div className="navbar-inner">
        {/* Brand */}
        <NavLink to="/" className="navbar-brand">
          <div className="navbar-logo">
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
              <defs>
                <linearGradient id="shield-grad" x1="0" y1="0" x2="32" y2="32">
                  <stop offset="0%" stopColor="#3b82f6" />
                  <stop offset="100%" stopColor="#8b5cf6" />
                </linearGradient>
              </defs>
              <path
                d="M16 2L4 8v8c0 7.73 5.12 14.96 12 16.73C22.88 30.96 28 23.73 28 16V8L16 2z"
                fill="url(#shield-grad)"
                opacity="0.2"
                stroke="url(#shield-grad)"
                strokeWidth="1.5"
              />
              <path
                d="M16 6L8 10v6c0 5.52 3.41 10.64 8 12 4.59-1.36 8-6.48 8-12v-6L16 6z"
                fill="none"
                stroke="url(#shield-grad)"
                strokeWidth="1.5"
              />
              {/* Waveform inside shield */}
              <path
                d="M11 16h2l1-3 1 6 1-4 1 3 1-2 1 1h2"
                fill="none"
                stroke="#60a5fa"
                strokeWidth="1.2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
          <span className="navbar-title">SpectraShield</span>
        </NavLink>

        {/* Navigation Links */}
        <div className="navbar-links">
          <NavLink
            to="/"
            end
            className={({ isActive }) => `navbar-link ${isActive ? 'navbar-link-active' : ''}`}
          >
            Home
          </NavLink>
          <NavLink
            to="/recorded"
            className={({ isActive }) => `navbar-link ${isActive ? 'navbar-link-active' : ''}`}
          >
            Recorded
          </NavLink>
          <NavLink
            to="/live"
            className={({ isActive }) => `navbar-link ${isActive ? 'navbar-link-active' : ''}`}
          >
            Live
          </NavLink>
        </div>

        {/* Status */}
        <div className="navbar-status">
          <span className={`status-dot ${getStatusClass()}`} />
          <span className="navbar-status-text">{getStatusText()}</span>
        </div>
      </div>
    </nav>
  );
}
