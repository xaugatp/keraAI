/**
 * Browser geolocation, honestly: starts EMPTY (no fake coordinates), only asks when `request()`
 * is called, reports the real accuracy and the fix's own timestamp.
 *
 * `request()` must be called from a user gesture (a button click): browsers — Safari especially —
 * ignore or auto-deny permission prompts that are not user-initiated. Geolocation also only works
 * on HTTPS or localhost.
 *
 * `position` is shaped so it can be spread straight into `predict()`:
 *   predict(key, { file, source, ...(position ?? {}) })
 */
import { useCallback, useEffect, useRef, useState } from 'react';

export type GeolocationStatus = 'idle' | 'requesting' | 'acquired' | 'denied' | 'unavailable';

export interface GeoPosition {
  latitude: number;
  longitude: number;
  /** Horizontal accuracy in metres, as reported by the device. */
  accuracyM: number;
  /** When the fix was taken (from `position.timestamp`, not "now"). */
  capturedAt: Date;
}

export interface UseGeolocationResult {
  status: GeolocationStatus;
  position: GeoPosition | null;
  /** Friendly explanation when `status` is 'denied' or 'unavailable'; null otherwise. */
  error: string | null;
  /** Ask for a fix. Call from a click handler. */
  request: () => void;
  /** Forget the current fix and go back to 'idle'. */
  clear: () => void;
}

const OPTIONS: PositionOptions = { enableHighAccuracy: true, timeout: 15_000, maximumAge: 0 };

export function useGeolocation(): UseGeolocationResult {
  const [status, setStatus] = useState<GeolocationStatus>('idle');
  const [position, setPosition] = useState<GeoPosition | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Latest request wins; results arriving after unmount / clear / a newer request are ignored.
  const requestId = useRef(0);

  useEffect(
    () => () => {
      requestId.current += 1;
    },
    [],
  );

  const request = useCallback(() => {
    const id = ++requestId.current;

    if (typeof navigator === 'undefined' || !navigator.geolocation) {
      setStatus('unavailable');
      setError('This device or browser cannot share a location.');
      return;
    }

    setStatus('requesting');
    setError(null);
    navigator.geolocation.getCurrentPosition(
      (fix) => {
        if (id !== requestId.current) return;
        setPosition({
          latitude: fix.coords.latitude,
          longitude: fix.coords.longitude,
          accuracyM: fix.coords.accuracy,
          capturedAt: new Date(fix.timestamp),
        });
        setStatus('acquired');
      },
      (failure) => {
        if (id !== requestId.current) return;
        setPosition(null);
        if (failure.code === failure.PERMISSION_DENIED) {
          setStatus('denied');
          setError('Location permission was denied. You can still analyse the photo without a location.');
        } else {
          setStatus('unavailable');
          setError(
            failure.code === failure.TIMEOUT
              ? 'Getting your location took too long. Try again, or continue without a location.'
              : "Couldn't determine your location. Try again, or continue without a location.",
          );
        }
      },
      OPTIONS,
    );
  }, []);

  const clear = useCallback(() => {
    requestId.current += 1;
    setPosition(null);
    setError(null);
    setStatus('idle');
  }, []);

  return { status, position, error, request, clear };
}
