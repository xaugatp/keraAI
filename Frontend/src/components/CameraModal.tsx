import React, { useState, useRef, useEffect, useCallback } from 'react';

interface CameraModalProps {
  isOpen: boolean;
  mode: 'plant' | 'leaf';
  onClose: () => void;
  /** `capturedAt` is an ISO-8601 timestamp taken at the moment the shutter was pressed. */
  onCapture: (file: File, capturedAt: string) => void;
  /** Decorative placeholder shown in the viewfinder before the live stream starts (never captured). */
  fallbackImage: string;
}

/** `HTMLMediaElement.HAVE_CURRENT_DATA`: the video has at least one real decoded frame to draw. */
const HAVE_CURRENT_DATA = 2;

export const CameraModal: React.FC<CameraModalProps> = ({
  isOpen,
  mode,
  onClose,
  onCapture,
  fallbackImage,
}) => {
  // Zoom & Lens State
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [activeLens, setActiveLens] = useState<'0.5x' | '1x' | '2x' | '3x'>('1x');

  // Exposure & Filter State
  const [exposureEv, setExposureEv] = useState<number>(0); // -1 to +1
  const [showGrid, setShowGrid] = useState<boolean>(true);
  const [isFlashOn, setIsFlashOn] = useState(false);
  const [isFlipped, setIsFlipped] = useState(false);

  // Focus simulation
  const [focusPoint, setFocusPoint] = useState<{ x: number; y: number; id: number } | null>(null);
  const [isFocusing, setIsFocusing] = useState(false);

  // Hardware Stream State
  const [isLiveStreamActive, setIsLiveStreamActive] = useState<boolean>(false);
  /** True once the `<video>` has a real decoded frame (readyState >= HAVE_CURRENT_DATA): only then can we capture. */
  const [videoReady, setVideoReady] = useState<boolean>(false);
  const [streamErrorNotice, setStreamErrorNotice] = useState<string | null>(null);

  // The captured photo, held as an object URL for preview plus the File/timestamp to hand back.
  const [capturedPreview, setCapturedPreview] = useState<string | null>(null);
  const [capturedFile, setCapturedFile] = useState<File | null>(null);
  const [capturedAt, setCapturedAt] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);

  const closeAudioContext = useCallback(() => {
    const ctx = audioCtxRef.current;
    if (ctx && ctx.state !== 'closed') {
      ctx.close().catch(() => {
        // nothing useful to do if closing fails
      });
    }
    audioCtxRef.current = null;
  }, []);

  const getAudioContext = useCallback((): AudioContext | null => {
    try {
      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      if (!AudioCtx) return null;
      if (!audioCtxRef.current || audioCtxRef.current.state === 'closed') {
        audioCtxRef.current = new AudioCtx();
      }
      return audioCtxRef.current;
    } catch {
      return null;
    }
  }, []);

  // Play synthetic camera shutter sound via Web Audio API (one shared, reusable AudioContext).
  const playSound = useCallback(
    (type: 'shutter' | 'focus') => {
      const ctx = getAudioContext();
      if (!ctx) return;
      try {
        if (type === 'shutter') {
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();
          osc.type = 'sine';
          osc.frequency.setValueAtTime(800, ctx.currentTime);
          osc.frequency.exponentialRampToValueAtTime(100, ctx.currentTime + 0.08);
          gain.gain.setValueAtTime(0.35, ctx.currentTime);
          gain.gain.linearRampToValueAtTime(0.01, ctx.currentTime + 0.08);
          osc.connect(gain);
          gain.connect(ctx.destination);
          osc.start();
          osc.stop(ctx.currentTime + 0.09);
        } else {
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();
          osc.type = 'sine';
          osc.frequency.setValueAtTime(1200, ctx.currentTime);
          gain.gain.setValueAtTime(0.08, ctx.currentTime);
          gain.gain.linearRampToValueAtTime(0.005, ctx.currentTime + 0.04);
          osc.connect(gain);
          gain.connect(ctx.destination);
          osc.start();
          osc.stop(ctx.currentTime + 0.05);
        }
      } catch {
        // Audio autoplay restrictions: safe to ignore.
      }
    },
    [getAudioContext],
  );

  const stopCameraStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch {
          // ignore
        }
      });
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsLiveStreamActive(false);
    setVideoReady(false);
  }, []);

  const startCameraStream = useCallback(async () => {
    stopCameraStream();
    setStreamErrorNotice(null);

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStreamErrorNotice("This browser can't access the camera. Use Upload Image instead.");
      return;
    }

    try {
      const constraints: MediaStreamConstraints = {
        video: {
          facingMode: isFlipped ? 'user' : { ideal: 'environment' },
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
        audio: false,
      };

      let stream: MediaStream | null = null;
      try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);
      } catch {
        stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });
      }

      if (stream && videoRef.current) {
        streamRef.current = stream;
        const video = videoRef.current;
        video.srcObject = stream;
        video.muted = true;
        video.setAttribute('playsinline', 'true');
        video.setAttribute('webkit-playsinline', 'true');

        video.onloadedmetadata = () => {
          video.play().then(() => {
            setIsLiveStreamActive(true);
          }).catch(() => {
            setIsLiveStreamActive(true);
          });
        };
        video.onloadeddata = () => {
          setVideoReady(true);
        };
      } else {
        setStreamErrorNotice('Could not start the camera. Use Upload Image instead.');
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'The camera is unavailable in this browser.';
      setStreamErrorNotice(msg);
      setIsLiveStreamActive(false);
    }
  }, [isFlipped, stopCameraStream]);

  useEffect(() => {
    if (isOpen) {
      setCapturedPreview(null);
      setCapturedFile(null);
      setCapturedAt(null);
      setZoomLevel(1);
      setActiveLens('1x');
      setExposureEv(0);
      startCameraStream();
    } else {
      stopCameraStream();
      setCapturedPreview(null);
      setCapturedFile(null);
      setCapturedAt(null);
      closeAudioContext();
    }
    return () => {
      stopCameraStream();
      closeAudioContext();
    };
  }, [isOpen, startCameraStream, stopCameraStream, closeAudioContext]);

  // Release the previous captured-photo object URL whenever it changes, and on unmount.
  useEffect(() => {
    return () => {
      if (capturedPreview) URL.revokeObjectURL(capturedPreview);
    };
  }, [capturedPreview]);

  // Handle tap-to-focus interactive animation
  const handleViewportClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (capturedPreview) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setFocusPoint({ x, y, id: Date.now() });
    setIsFocusing(true);
    playSound('focus');

    setTimeout(() => {
      setIsFocusing(false);
    }, 900);
  };

  // Change lens presets
  const handleLensSelect = (lensVal: '0.5x' | '1x' | '2x' | '3x') => {
    setActiveLens(lensVal);
    if (lensVal === '0.5x') setZoomLevel(0.85);
    else if (lensVal === '1x') setZoomLevel(1);
    else if (lensVal === '2x') setZoomLevel(1.6);
    else if (lensVal === '3x') setZoomLevel(2.2);
    playSound('focus');
  };

  // Interactive Shutter Capture: draws the REAL video frame at its native resolution to a canvas,
  // then encodes it as a JPEG File. No capture is possible until a real frame has arrived
  // (the shutter button is disabled until then), so there is no "blank navy" or stock-photo
  // fallback frame to draw.
  const handleCaptureClick = () => {
    if (!videoReady || !videoRef.current) return;
    const shutterPressedAt = Date.now();
    setIsFlashOn(true);
    playSound('shutter');

    setTimeout(() => {
      setIsFlashOn(false);

      const video = videoRef.current;
      if (!video || video.readyState < HAVE_CURRENT_DATA || !video.videoWidth || !video.videoHeight) {
        setStreamErrorNotice('Lost the camera frame before the photo could be captured. Please try again.');
        return;
      }

      try {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        if (!ctx) throw new Error('2D canvas context unavailable');

        // Apply exposure / brightness (cosmetic, matches the live preview's filter).
        const brightnessFactor = 1 + exposureEv * 0.25;
        ctx.filter = `brightness(${brightnessFactor})`;

        ctx.save();
        if (isFlipped) {
          ctx.translate(canvas.width, 0);
          ctx.scale(-1, 1);
        }
        // Cosmetic zoom: crop/scale the drawn frame, same as the live preview.
        const w = canvas.width * zoomLevel;
        const h = canvas.height * zoomLevel;
        const ox = (canvas.width - w) / 2;
        const oy = (canvas.height - h) / 2;
        ctx.drawImage(video, ox, oy, w, h);
        ctx.restore();

        canvas.toBlob(
          (blob) => {
            if (!blob) {
              setStreamErrorNotice("Couldn't capture a photo from the camera. Please try again.");
              return;
            }
            const iso = new Date(shutterPressedAt).toISOString();
            const file = new File([blob], `camera-capture-${shutterPressedAt}.jpg`, { type: 'image/jpeg' });
            setCapturedFile(file);
            setCapturedAt(iso);
            setCapturedPreview(URL.createObjectURL(blob));
          },
          'image/jpeg',
          0.92,
        );
      } catch {
        setStreamErrorNotice("Couldn't capture a photo from the camera. Please try again.");
      }
    }, 120);
  };

  const confirmCapturedPhoto = () => {
    if (capturedFile && capturedAt) {
      onCapture(capturedFile, capturedAt);
      onClose();
    }
  };

  const retakePhoto = () => {
    setCapturedPreview(null);
    setCapturedFile(null);
    setCapturedAt(null);
  };

  if (!isOpen) return null;

  const currentBrightness = 1 + exposureEv * 0.25;

  return (
    <div className="fixed inset-0 z-50 bg-[#090d16]/95 backdrop-blur-xl flex flex-col justify-between items-center p-3 sm:p-5 select-none animate-fade-in font-sans">
      {/* 1. TOP STATUS & CONTROLS HUD */}
      <div className="w-full max-w-5xl flex items-center justify-between text-white py-2 px-3 z-30 flex-wrap gap-2">
        {/* Left: Mode Title & Sensor Status */}
        <div className="flex items-center gap-2.5">
          <div className="relative flex items-center justify-center">
            <span
              className={`w-3 h-3 rounded-full ${
                isLiveStreamActive ? 'bg-[#85f8c4] animate-pulse' : 'bg-[#00a86b]'
              }`}
            />
            <span className="w-3 h-3 rounded-full bg-[#85f8c4] absolute animate-ping opacity-60" />
          </div>

          <div className="flex flex-col">
            <span className="font-mono text-xs sm:text-sm font-extrabold tracking-wider uppercase text-white flex items-center gap-1.5">
              <span>{mode === 'plant' ? 'Banana Plant Viewfinder' : 'Leaf Macro Diagnostic Camera'}</span>
            </span>
            <span className="text-[10px] font-mono text-[#85f8c4]">
              {isLiveStreamActive ? 'Live Sensor Feed' : 'Calibrated Macro Sensor'}
            </span>
          </div>
        </div>

        {/* Right: Camera Action Buttons */}
        <div className="flex items-center gap-2 sm:gap-2.5 flex-wrap">
          {/* Grid Toggle */}
          <button
            type="button"
            onClick={() => setShowGrid(!showGrid)}
            className={`w-9 h-9 rounded-full flex items-center justify-center text-xs font-mono transition-all cursor-pointer ${
              showGrid ? 'bg-white/25 text-[#85f8c4] border border-[#85f8c4]/50' : 'bg-white/10 text-white/70'
            }`}
            title="Toggle Rule-of-Thirds Composition Grid"
          >
            <span className="material-symbols-outlined text-[18px]">grid_4x4</span>
          </button>

          {/* Close Camera */}
          <button
            type="button"
            onClick={onClose}
            className="w-9 h-9 rounded-full bg-white/10 hover:bg-white/20 text-white flex items-center justify-center transition-colors cursor-pointer"
            title="Close Viewfinder"
          >
            <span className="material-symbols-outlined text-[20px]">close</span>
          </button>
        </div>
      </div>

      {/* 2. CENTER VIEWFINDER VIEWPORT */}
      <div
        ref={viewportRef}
        onClick={handleViewportClick}
        className="relative w-full max-w-5xl flex-1 max-h-[74dvh] rounded-3xl overflow-hidden bg-black flex items-center justify-center border-2 border-white/20 shadow-2xl cursor-crosshair group select-none"
      >
        {/* Shutter Flash Effect */}
        {isFlashOn && (
          <div className="absolute inset-0 bg-white z-50 transition-opacity duration-100 pointer-events-none" />
        )}

        {/* Live Video Feed Element (if WebRTC is active) */}
        <video
          ref={videoRef}
          playsInline
          autoPlay
          muted
          className={`w-full h-full object-cover transition-transform duration-300 pointer-events-none ${
            isLiveStreamActive && !capturedPreview ? 'block' : 'hidden'
          }`}
          style={{
            transform: `scale(${zoomLevel}) ${isFlipped ? 'scaleX(-1)' : ''}`,
            filter: `brightness(${currentBrightness})`,
          }}
        />

        {/* Placeholder specimen graphic (not a capture source) while the stream isn't live, or the frozen capture preview. */}
        {(!isLiveStreamActive || capturedPreview) && (
          <div className="relative w-full h-full flex items-center justify-center bg-[#090d16] overflow-hidden">
            <img
              alt={capturedPreview ? 'Captured specimen preview' : 'Camera viewfinder placeholder'}
              src={capturedPreview || fallbackImage}
              className={`w-full h-full object-cover transition-all duration-300 pointer-events-none ${
                capturedPreview ? 'object-contain' : ''
              }`}
              style={{
                transform: capturedPreview ? 'none' : `scale(${zoomLevel}) ${isFlipped ? 'scaleX(-1)' : ''}`,
                filter: capturedPreview ? 'none' : `brightness(${currentBrightness})`,
              }}
            />
          </div>
        )}

        {/* Review Snapshot Confirmation Banner / Stream Error Notice / Live Hint */}
        {capturedPreview ? (
          <div className="absolute top-5 left-5 bg-black/85 backdrop-blur-md px-4 py-2 rounded-2xl border border-[#85f8c4]/60 text-white font-mono text-xs flex items-center gap-2.5 shadow-xl animate-fade-in z-40">
            <span className="w-2.5 h-2.5 rounded-full bg-[#85f8c4] animate-pulse" />
            <span className="font-bold text-white">Specimen Frame Frozen</span>
            <span className="text-[#85f8c4]">&bull; Ready for Diagnostics</span>
          </div>
        ) : streamErrorNotice ? (
          <div className="absolute top-4 inset-x-0 flex justify-center pointer-events-none z-30 px-4">
            <div className="bg-[#ba1a1a]/90 backdrop-blur-md px-4 py-2 rounded-2xl border border-white/30 text-white font-mono text-xs flex items-center gap-2 shadow-xl max-w-md text-center">
              <span className="material-symbols-outlined text-[16px]">warning</span>
              <span>{streamErrorNotice}</span>
            </div>
          </div>
        ) : (
          /* Live Sensor Indicator Badge (Non-blocking, sleek, interactive) */
          <div className="absolute top-4 inset-x-0 flex justify-center pointer-events-none z-30">
            <div className="bg-black/75 backdrop-blur-md px-4 py-1.5 rounded-full border border-white/20 text-white font-mono text-xs flex items-center gap-2 shadow-lg">
              <span className="material-symbols-outlined text-[#85f8c4] text-[16px]">touch_app</span>
              <span>Tap anywhere to autofocus &bull; Frame specimen inside brackets</span>
            </div>
          </div>
        )}

        {/* Rule of Thirds Agricultural Calibration Grid */}
        {showGrid && !capturedPreview && (
          <div className="absolute inset-0 pointer-events-none z-10 grid grid-cols-3 grid-rows-3">
            <div className="border-r border-b border-white/15" />
            <div className="border-r border-b border-white/15" />
            <div className="border-b border-white/15" />
            <div className="border-r border-b border-white/15" />
            <div className="border-r border-b border-white/15" />
            <div className="border-b border-white/15" />
            <div className="border-r border-white/15" />
            <div className="border-r border-white/15" />
            <div />
          </div>
        )}

        {/* Viewfinder Target Reticle & Corner Brackets */}
        {!capturedPreview && (
          <div className="absolute inset-0 pointer-events-none flex flex-col justify-between p-5 sm:p-8 z-20">
            {/* Top Brackets */}
            <div className="flex justify-between items-start">
              <div className="w-8 h-8 border-t-2 border-l-2 border-[#85f8c4] drop-shadow-[0_0_8px_rgba(133,248,196,0.6)]" />
              <div className="w-8 h-8 border-t-2 border-r-2 border-[#85f8c4] drop-shadow-[0_0_8px_rgba(133,248,196,0.6)]" />
            </div>

            {/* Center Dynamic Crosshair */}
            <div className="self-center flex items-center justify-center relative w-40 h-40">
              <div className="w-8 h-0.5 bg-[#85f8c4]/80 absolute" />
              <div className="h-8 w-0.5 bg-[#85f8c4]/80 absolute" />
              <div className="w-20 h-20 rounded-full border border-[#85f8c4]/30 absolute animate-pulse" />
              <span className="absolute bottom-2 font-mono text-[9px] text-[#85f8c4] tracking-widest uppercase">
                {activeLens} &bull; {mode.toUpperCase()}
              </span>
            </div>

            {/* Bottom Brackets */}
            <div className="flex justify-between items-end">
              <div className="w-8 h-8 border-b-2 border-l-2 border-[#85f8c4] drop-shadow-[0_0_8px_rgba(133,248,196,0.6)]" />
              <div className="w-8 h-8 border-b-2 border-r-2 border-[#85f8c4] drop-shadow-[0_0_8px_rgba(133,248,196,0.6)]" />
            </div>
          </div>
        )}

        {/* Interactive Tap-To-Focus Ring */}
        {focusPoint && !capturedPreview && (
          <div
            key={focusPoint.id}
            style={{
              left: `${focusPoint.x}px`,
              top: `${focusPoint.y}px`,
            }}
            className="absolute -translate-x-1/2 -translate-y-1/2 pointer-events-none z-30 flex flex-col items-center"
          >
            <div
              className={`w-16 h-16 rounded-xl border-2 transition-all duration-300 ${
                isFocusing
                  ? 'border-[#85f8c4] scale-90 shadow-[0_0_16px_#85f8c4]'
                  : 'border-[#85f8c4]/70 scale-100'
              }`}
            >
              <div className="absolute inset-x-0 top-1/2 h-[1px] bg-[#85f8c4]/60" />
              <div className="absolute inset-y-0 left-1/2 w-[1px] bg-[#85f8c4]/60" />
            </div>
            <span className="font-mono text-[9px] text-[#85f8c4] font-bold mt-1 bg-black/75 px-1.5 py-0.5 rounded">
              AF: LOCKED
            </span>
          </div>
        )}
      </div>

      {/* 3. BOTTOM INTERACTIVE DOCK & SHUTTER CONTROLS */}
      <div className="w-full max-w-5xl py-3 px-4 flex items-center justify-between text-white z-30 gap-4">
        {capturedPreview ? (
          /* Snapshot Review Actions */
          <div className="w-full flex items-center justify-center gap-4 flex-wrap">
            <button
              type="button"
              onClick={retakePhoto}
              className="px-6 py-3 rounded-2xl bg-white/15 hover:bg-white/25 text-white font-semibold text-sm transition-all flex items-center gap-2 border border-white/20 cursor-pointer active:scale-95 shadow-md"
            >
              <span className="material-symbols-outlined text-[20px]">refresh</span>
              <span>Retake Photo</span>
            </button>

            <button
              type="button"
              onClick={confirmCapturedPhoto}
              className="px-8 py-3 rounded-2xl bg-gradient-to-r from-[#006948] to-[#00855d] hover:brightness-110 text-white font-bold text-sm shadow-xl flex items-center gap-2 transition-all cursor-pointer active:scale-95 border-2 border-[#85f8c4]"
            >
              <span className="material-symbols-outlined text-[22px]">check_circle</span>
              <span>Analyze This Specimen</span>
            </button>
          </div>
        ) : (
          /* Live Shutter, Zoom & Exposure Dock */
          <div className="w-full flex items-center justify-between">
            {/* Left: Lens Selectors */}
            <div className="flex items-center gap-1 bg-black/60 backdrop-blur-md rounded-full p-1 border border-white/20 shadow-md">
              {(['0.5x', '1x', '2x', '3x'] as const).map((l) => (
                <button
                  key={l}
                  type="button"
                  onClick={() => handleLensSelect(l)}
                  className={`px-3 py-1.5 rounded-full font-mono text-xs font-bold transition-all cursor-pointer ${
                    activeLens === l
                      ? 'bg-[#85f8c4] text-[#002114] shadow-sm scale-105'
                      : 'text-white/70 hover:text-white'
                  }`}
                >
                  {l}
                </button>
              ))}
            </div>

            {/* Center: Prominent Interactive Shutter Button */}
            <div className="flex items-center justify-center">
              <button
                type="button"
                onClick={handleCaptureClick}
                disabled={!videoReady}
                title={videoReady ? 'Capture Specimen Image' : 'Waiting for the camera…'}
                className="relative group p-2 rounded-full bg-white/10 hover:bg-white/20 transition-all cursor-pointer active:scale-90 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-white/10"
              >
                {videoReady && (
                  <span className="absolute inset-0 rounded-full bg-[#85f8c4] opacity-40 animate-ping pointer-events-none" />
                )}
                <div className="w-16 h-16 sm:w-18 sm:h-18 rounded-full bg-gradient-to-tr from-[#006948] to-[#00855d] flex items-center justify-center text-white shadow-[0_0_24px_rgba(0,105,72,0.6)] group-hover:scale-105 transition-transform border-4 border-white">
                  <span className="material-symbols-outlined text-[32px]">photo_camera</span>
                </div>
              </button>
            </div>

            {/* Right: Exposure Adjustment & Camera Flip */}
            <div className="flex items-center gap-2">
              {/* Exposure toggle EV */}
              <button
                type="button"
                onClick={() => setExposureEv((prev) => (prev >= 1 ? -1 : prev + 1))}
                className="px-2.5 py-1.5 rounded-full bg-black/60 backdrop-blur-md text-white/90 hover:text-white font-mono text-xs border border-white/20 flex items-center gap-1 cursor-pointer transition-colors"
                title="Adjust Exposure (EV)"
              >
                <span className="material-symbols-outlined text-[15px] text-[#85f8c4]">exposure</span>
                <span>{exposureEv >= 0 ? `+${exposureEv}` : exposureEv}</span>
              </button>

              {/* Flip camera */}
              <button
                type="button"
                onClick={() => setIsFlipped(!isFlipped)}
                className="w-10 h-10 rounded-full bg-black/60 backdrop-blur-md hover:bg-white/20 flex items-center justify-center text-white border border-white/20 transition-colors cursor-pointer"
                title="Flip Camera Orientation"
              >
                <span className="material-symbols-outlined text-[20px]">cameraswitch</span>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
