import { useState, useRef, useCallback } from 'react';
import './FileDropZone.css';

const ALLOWED_TYPES = ['.wav', '.mp3', '.flac', '.m4a', '.ogg'];

export default function FileDropZone({ onFileSelect, disabled }) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const inputRef = useRef(null);

  const validateFile = (file) => {
    if (!file) return null;
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!ALLOWED_TYPES.includes(ext)) {
      return `Unsupported format: ${ext}. Accepted: ${ALLOWED_TYPES.join(', ')}`;
    }
    if (file.size > 100 * 1024 * 1024) {
      return 'File too large. Maximum: 100MB';
    }
    return null;
  };

  const handleFile = useCallback(
    (file) => {
      const error = validateFile(file);
      if (error) {
        alert(error);
        return;
      }
      setSelectedFile(file);
      onFileSelect(file);
    },
    [onFileSelect]
  );

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragging(false);
      if (disabled) return;
      const file = e.dataTransfer?.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile, disabled]
  );

  const handleDragOver = (e) => {
    e.preventDefault();
    if (!disabled) setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleInputChange = (e) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const handleClick = () => {
    if (!disabled) inputRef.current?.click();
  };

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div
      className={`dropzone glass-card ${isDragging ? 'dropzone-active' : ''} ${disabled ? 'dropzone-disabled' : ''}`}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onClick={handleClick}
      role="button"
      tabIndex={0}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ALLOWED_TYPES.join(',')}
        onChange={handleInputChange}
        className="dropzone-input"
        disabled={disabled}
      />

      <div className="dropzone-content">
        {selectedFile ? (
          <>
            <div className="dropzone-file-icon">
              <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                <rect x="8" y="4" width="32" height="40" rx="4" fill="rgba(59,130,246,0.15)" stroke="#3b82f6" strokeWidth="1.5" />
                <path d="M16 24h16M16 30h10" stroke="#60a5fa" strokeWidth="1.5" strokeLinecap="round" />
                <circle cx="34" cy="14" r="6" fill="#10b981" />
                <path d="M31 14l2 2 4-4" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            <p className="dropzone-filename">{selectedFile.name}</p>
            <p className="dropzone-filesize">{formatSize(selectedFile.size)}</p>
          </>
        ) : (
          <>
            <div className="dropzone-upload-icon">
              <svg width="56" height="56" viewBox="0 0 56 56" fill="none">
                <circle cx="28" cy="28" r="27" stroke="url(#dz-grad)" strokeWidth="1.5" strokeDasharray="4 4" />
                <path
                  d="M28 18v16M22 24l6-6 6 6"
                  stroke="#60a5fa"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M18 36h20"
                  stroke="#3b82f6"
                  strokeWidth="2"
                  strokeLinecap="round"
                />
                <defs>
                  <linearGradient id="dz-grad" x1="0" y1="0" x2="56" y2="56">
                    <stop offset="0%" stopColor="#3b82f6" />
                    <stop offset="100%" stopColor="#8b5cf6" />
                  </linearGradient>
                </defs>
              </svg>
            </div>
            <p className="dropzone-title">
              {isDragging ? 'Drop your audio file' : 'Drag & drop audio file here'}
            </p>
            <p className="dropzone-subtitle">or click to browse</p>
            <div className="dropzone-formats">
              {ALLOWED_TYPES.map((ext) => (
                <span key={ext} className="dropzone-format-tag">{ext}</span>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
