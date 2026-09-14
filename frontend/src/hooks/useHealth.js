import { useState, useEffect, useRef, useCallback } from 'react';
import { checkHealth } from '../services/api';

/**
 * useHealth — Polls /api/health every 30 seconds.
 * Returns { isConnected, modelLoaded, device, error, refetch }
 */
export function useHealth() {
  const [state, setState] = useState({
    isConnected: false,
    modelLoaded: false,
    device: null,
    error: null,
  });

  const intervalRef = useRef(null);

  const refetch = useCallback(async () => {
    try {
      const data = await checkHealth();
      setState({
        isConnected: true,
        modelLoaded: data.model_loaded,
        device: data.device,
        error: null,
      });
    } catch {
      setState((prev) => ({
        ...prev,
        isConnected: false,
        error: 'Cannot reach backend',
      }));
    }
  }, []);

  useEffect(() => {
    refetch();
    intervalRef.current = setInterval(refetch, 30000);
    return () => clearInterval(intervalRef.current);
  }, [refetch]);

  return { ...state, refetch };
}
