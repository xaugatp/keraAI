import React, { useState, useRef, useEffect, useCallback } from 'react';

interface CameraModalProps {
  isOpen: boolean;
  mode: 'plant' | 'leaf';
  onClose: () => void;
  onCapture: (capturedImage: string) => void;
  fallbackImage: string;
}

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
  // The notice text is set below but not rendered yet; F2 (spec §2, CameraModal) will display it.
  const [, setStreamErrorNotice] = useState<string | null>(null);
  const [capturedPreview, setCapturedPreview] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const nativeInputRef = useRef<HTMLInputElement | null>(null);
  const viewportRef = useRef<HTMLDivElement | null>(null);

  // Play synthetic camera shutter sound via Web Audio API
  const playSound = (type: 'shutter' | 'focus') => {
    try {
      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();

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
      } else if (type === 'focus') {
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
      // Audio autoplay restrictions safe ignore
    }
  };

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
  }, []);

  const startCameraStream = useCallback(async () => {
    stopCameraStream();
    setStreamErrorNotice(null);

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStreamErrorNotice('Device WebRTC restricted in preview iframe. Using Interactive Sensor Mode.');
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
      } else {
        setStreamErrorNotice('Sensor simulated in studio view');
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'WebRTC unavailable in this browser frame';
      setStreamErrorNotice(msg);
      setIsLiveStreamActive(false);
    }
  }, [isFlipped, stopCameraStream]);

  useEffect(() => {
    if (isOpen) {
      setCapturedPreview(null);
      setZoomLevel(1);
      setActiveLens('1x');
      setExposureEv(0);
      startCameraStream();
    } else {
      stopCameraStream();
      setCapturedPreview(null);
    }
    return () => {
      stopCameraStream();
    };
  }, [isOpen, startCameraStream, stopCameraStream]);

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

  // Interactive Shutter Capture
  const handleCaptureClick = () => {
    setIsFlashOn(true);
    playSound('shutter');

    setTimeout(() => {
      setIsFlashOn(false);

      try {
        const canvas = document.createElement('canvas');
        canvas.width = 1600;
        canvas.height = 1200;
        const ctx = canvas.getContext('2d');

        if (ctx) {
          // Fill canvas background
          ctx.fillStyle = '#131b2e';
          ctx.fillRect(0, 0, canvas.width, canvas.height);

          // Apply exposure / brightness
          const brightnessFactor = 1 + exposureEv * 0.25;
          ctx.filter = `brightness(${brightnessFactor})`;

          if (isLiveStreamActive && videoRef.current && videoRef.current.readyState >= 2) {
            const video = videoRef.current;
            ctx.save();
            if (isFlipped) {
              ctx.translate(canvas.width, 0);
              ctx.scale(-1, 1);
            }
            // Scale according to zoomLevel
            const w = canvas.width * zoomLevel;
            const h = canvas.height * zoomLevel;
            const ox = (canvas.width - w) / 2;
            const oy = (canvas.height - h) / 2;
            ctx.drawImage(video, ox, oy, w, h);
            ctx.restore();
          } else {
            // High-res image specimen source
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.src = fallbackImage;

            ctx.save();
            if (isFlipped) {
              ctx.translate(canvas.width, 0);
              ctx.scale(-1, 1);
            }
            const w = canvas.width * zoomLevel;
            const h = canvas.height * zoomLevel;
            const ox = (canvas.width - w) / 2;
            const oy = (canvas.height - h) / 2;
            ctx.drawImage(img, ox, oy, w, h);
            ctx.restore();
          }

          const dataUrl = canvas.toDataURL('image/jpeg', 0.94);
          setCapturedPreview(dataUrl);
          return;
        }
      } catch {
        // Fallback to active specimen image
      }

      setCapturedPreview(fallbackImage);
    }, 120);
  };

  // Direct Native Mobile Camera launcher
  const handleNativeCameraCapture = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          setCapturedPreview(event.target.result as string);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const confirmCapturedPhoto = () => {
    if (capturedPreview) {
      onCapture(capturedPreview);
      onClose();
    }
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
              {isLiveStreamActive ? 'Live Sensor Feed' : 'Calibrated Macro Sensor'} &bull; 60 FPS
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

      {/* Hidden Native Phone Camera File Input */}
      <input
        ref={nativeInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="sr-only"
        onChange={handleNativeCameraCapture}
      />

      {/* 2. CENTER VIEWFINDER VIEWPORT */}
      <div
        ref={viewportRef}
        onClick={handleViewportClick}
        className="relative w-full max-w-5xl flex-1 max-h-[74vh] rounded-3xl overflow-hidden bg-black flex items-center justify-center border-2 border-white/20 shadow-2xl cursor-crosshair group select-none"
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

        {/* High-Resolution Interactive Calibrated Specimen (when stream is not active or as preview) */}
        {(!isLiveStreamActive || capturedPreview) && (
          <div className="relative w-full h-full flex items-center justify-center bg-[#090d16] overflow-hidden">
            <img
              alt="Interactive camera viewfinder specimen"
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

        {/* Review Snapshot Confirmation Banner */}
        {capturedPreview ? (
          <div className="absolute top-5 left-5 bg-black/85 backdrop-blur-md px-4 py-2 rounded-2xl border border-[#85f8c4]/60 text-white font-mono text-xs flex items-center gap-2.5 shadow-xl animate-fade-in z-40">
            <span className="w-2.5 h-2.5 rounded-full bg-[#85f8c4] animate-pulse" />
            <span className="font-bold text-white">Specimen Frame Frozen</span>
            <span className="text-[#85f8c4]">&bull; Ready for Diagnostics</span>
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
              onClick={() => setCapturedPreview(null)}
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
                className="relative group p-2 rounded-full bg-white/10 hover:bg-white/20 transition-all cursor-pointer active:scale-90"
                title="Capture Specimen Image"
              >
                <span className="absolute inset-0 rounded-full bg-[#85f8c4] opacity-40 animate-ping pointer-events-none" />
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
