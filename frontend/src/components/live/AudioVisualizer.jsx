import { useMemo } from 'react';
import './AudioVisualizer.css';

export default function AudioVisualizer({ audioLevel, isActive }) {
  const bars = useMemo(() => {
    const count = 32;
    return Array.from({ length: count }, (_, i) => {
      // Create a wave pattern based on audio level
      const position = i / count;
      const centerDistance = Math.abs(position - 0.5) * 2;
      const base = isActive ? Math.max(0.08, audioLevel * 8 * (1 - centerDistance * 0.6)) : 0.08;
      // Add some randomness for natural feel
      const height = Math.min(1, base + (isActive ? Math.random() * audioLevel * 3 : 0));
      return height;
    });
  }, [audioLevel, isActive]);

  return (
    <div className={`audio-visualizer ${isActive ? 'audio-visualizer-active' : ''}`}>
      <div className="audio-bars">
        {bars.map((height, i) => (
          <div
            key={i}
            className="audio-bar"
            style={{
              height: `${Math.max(8, height * 100)}%`,
              animationDelay: `${i * 30}ms`,
            }}
          />
        ))}
      </div>
    </div>
  );
}
