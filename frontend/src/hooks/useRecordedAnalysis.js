import { useState, useCallback } from 'react';
import { predictRecorded } from '../services/api';

/**
 * useRecordedAnalysis — Manages file upload → prediction workflow.
 *
 * States: idle → uploading → done | error
 * Returns { upload, result, error, isLoading, reset }
 */
export function useRecordedAnalysis(initialResult = null) {
  const [result, setResult] = useState(initialResult);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const upload = useCallback(async (file) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await predictRecorded(file);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Analysis failed');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
    setIsLoading(false);
  }, []);

  return { upload, result, error, isLoading, reset };
}
