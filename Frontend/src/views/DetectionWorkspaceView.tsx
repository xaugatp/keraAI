import React, { useState, useRef, useEffect } from 'react';
import { ViewTab } from '../types';
import { ASSETS } from '../data/mockData';
import { CameraModal } from '../components/CameraModal';
import { ProcessingModal } from '../components/ProcessingModal';

interface DetectionWorkspaceViewProps {
  onNavigate: (tab: ViewTab) => void;
  onSelectSample?: (sampleType: 'positive' | 'negative' | 'custom', customImg?: string) => void;
  onGpsChange?: (coords: { latitude: number; longitude: number; accuracy?: number | null }) => void;
}

export const DetectionWorkspaceView: React.FC<DetectionWorkspaceViewProps> = ({
  onNavigate,
  onSelectSample,
  onGpsChange,
}) => {
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const nativeCameraInputRef = useRef<HTMLInputElement | null>(null);

  // Field GPS Geolocation state
  const [gpsLocation, setGpsLocation] = useState<{
    latitude: number;
    longitude: number;
    accuracy: number | null;
    timestamp: string;
    status: 'idle' | 'locating' | 'acquired' | 'error';
    errorMessage?: string;
  }>({
    latitude: 27.71724,
    longitude: 85.32402,
    accuracy: 4.2,
    timestamp: 'Field Block 4A',
    status: 'acquired',
  });
  const [isEditingGps, setIsEditingGps] = useState(false);
  const [customLat, setCustomLat] = useState('27.717240');
  const [customLng, setCustomLng] = useState('85.324020');

  useEffect(() => {
    // Attempt auto-acquisition of coordinates if available
    acquireCurrentLocation(false);
  }, []);

  const acquireCurrentLocation = (explicitClick = true) => {
    if (!navigator.geolocation) {
      if (explicitClick) {
        setGpsLocation((prev) => ({
          ...prev,
          status: 'error',
          errorMessage: 'Geolocation is not supported by your browser',
        }));
      }
      return;
    }

    setGpsLocation((prev) => ({ ...prev, status: 'locating', errorMessage: undefined }));

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const lat = Number(position.coords.latitude.toFixed(6));
        const lng = Number(position.coords.longitude.toFixed(6));
        const acc = position.coords.accuracy ? Number(position.coords.accuracy.toFixed(1)) : null;
        setGpsLocation({
          latitude: lat,
          longitude: lng,
          accuracy: acc,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          status: 'acquired',
        });
        setCustomLat(lat.toString());
        setCustomLng(lng.toString());
        onGpsChange?.({ latitude: lat, longitude: lng, accuracy: acc });
      },
      (error) => {
        let msg = 'Unable to retrieve location';
        if (error.code === error.PERMISSION_DENIED) {
          msg = 'Location permission not granted. Using plantation default.';
        } else if (error.code === error.POSITION_UNAVAILABLE) {
          msg = 'GPS signal unavailable. Using plantation default.';
        } else if (error.code === error.TIMEOUT) {
          msg = 'Location timed out. Using plantation default.';
        }
        setGpsLocation((prev) => ({
          ...prev,
          status: prev.status === 'acquired' ? 'acquired' : 'error',
          errorMessage: explicitClick ? msg : undefined,
        }));
      },
      {
        enableHighAccuracy: true,
        timeout: 8000,
        maximumAge: 30000,
      }
    );
  };

  const handleManualGpsSave = (e: React.FormEvent) => {
    e.preventDefault();
    const lat = parseFloat(customLat);
    const lng = parseFloat(customLng);
    if (!isNaN(lat) && !isNaN(lng)) {
      setGpsLocation({
        latitude: lat,
        longitude: lng,
        accuracy: null,
        timestamp: 'Manual Tag',
        status: 'acquired',
      });
      onGpsChange?.({ latitude: lat, longitude: lng, accuracy: null });
      setIsEditingGps(false);
    }
  };

  const startAnalysis = (sampleType: 'positive' | 'negative' | 'custom', customImg?: string) => {
    onSelectSample?.(sampleType, customImg);
    setIsProcessing(true);
  };

  const handleProcessingComplete = () => {
    setIsProcessing(false);
    onNavigate('plant-result');
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const isSuspectedNegative =
        file.name.toLowerCase().includes('succulent') ||
        file.name.toLowerCase().includes('houseplant') ||
        file.name.toLowerCase().includes('negative');
      const reader = new FileReader();
      reader.onload = (event) => {
        startAnalysis(isSuspectedNegative ? 'negative' : 'custom', event.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleNativeMobileCapture = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        startAnalysis('positive', event.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="w-full max-w-7xl mx-auto px-6 lg:px-12 py-10 flex flex-col gap-8">
      {/* Hidden Universal Native Camera Input for iPhone & Android */}
      <input
        ref={nativeCameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="sr-only"
        onChange={handleNativeMobileCapture}
      />

      {/* Full-Screen Mobile-Optimized Camera Modal */}
      <CameraModal
        isOpen={isCameraOpen}
        mode="plant"
        fallbackImage={ASSETS.bananaFieldCalib}
        onClose={() => setIsCameraOpen(false)}
        onCapture={(img) => {
          setIsCameraOpen(false);
          startAnalysis('positive', img);
        }}
      />

      {/* Asynchronous Processing Modal with 4-stage checklist */}
      <ProcessingModal
        isOpen={isProcessing}
        mode="plant"
        onComplete={handleProcessingComplete}
      />

      {/* Header - strictly "Plant Detection" as requested */}
      <header className="flex flex-col gap-2 max-w-3xl">
        <div className="flex items-center gap-2 text-[#3d4a42] text-xs font-semibold tracking-wider uppercase">
          <button
            onClick={() => onNavigate('home')}
            className="hover:text-[#006948] transition-colors cursor-pointer"
          >
            Home
          </button>
          <span className="material-symbols-outlined text-[14px]">chevron_right</span>
          <span className="text-[#006948] font-bold">Plant Detection</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-[#131b2e] tracking-tight">
          Plant Detection
        </h1>
        <p className="text-base text-[#3d4a42] leading-relaxed">
          Upload an image or use your camera to analyze a banana plant.
        </p>
      </header>

      {/* Field Location (Latitude & Longitude) Feature Bar */}
      <div className="bg-white p-5 rounded-2xl border border-[#dae2fd] shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-[#e2e7ff] text-[#006948] flex items-center justify-center shrink-0">
            <span className="material-symbols-outlined text-[22px]">location_on</span>
          </div>
          <div className="flex flex-col">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs uppercase font-bold text-[#3d4a42] tracking-wider">
                Field GPS Coordinates
              </span>
              {gpsLocation.status === 'locating' ? (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-[#eaedff] text-[#006948] font-mono text-[10px] font-bold">
                  <span className="material-symbols-outlined text-[12px] animate-spin">refresh</span>
                  Acquiring...
                </span>
              ) : gpsLocation.status === 'acquired' ? (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-[#85f8c4]/40 text-[#006948] font-mono text-[10px] font-bold">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#006948]" />
                  GPS Locked {gpsLocation.accuracy ? `(±${gpsLocation.accuracy}m)` : ''}
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-[#ffdad6] text-[#ba1a1a] font-mono text-[10px] font-bold">
                  Fallback Location
                </span>
              )}
            </div>

            {/* Coordinates Display */}
            {!isEditingGps ? (
              <div className="flex items-center gap-4 mt-1 font-mono text-xs sm:text-sm">
                <span className="text-[#131b2e]">
                  <strong className="text-[#3d4a42] font-semibold">Lat:</strong>{' '}
                  <span className="font-bold text-[#006948]">{gpsLocation.latitude.toFixed(6)}°</span>
                </span>
                <span className="text-[#bccac0]">&bull;</span>
                <span className="text-[#131b2e]">
                  <strong className="text-[#3d4a42] font-semibold">Long:</strong>{' '}
                  <span className="font-bold text-[#006948]">{gpsLocation.longitude.toFixed(6)}°</span>
                </span>
                <span className="text-xs text-[#3d4a42] hidden lg:inline">
                  ({gpsLocation.timestamp})
                </span>
              </div>
            ) : (
              <form onSubmit={handleManualGpsSave} className="flex items-center gap-2 mt-1.5 flex-wrap">
                <input
                  type="text"
                  placeholder="Latitude"
                  value={customLat}
                  onChange={(e) => setCustomLat(e.target.value)}
                  className="px-2.5 py-1 rounded-lg border border-[#dae2fd] text-xs font-mono w-28 bg-[#f2f3ff]"
                />
                <input
                  type="text"
                  placeholder="Longitude"
                  value={customLng}
                  onChange={(e) => setCustomLng(e.target.value)}
                  className="px-2.5 py-1 rounded-lg border border-[#dae2fd] text-xs font-mono w-28 bg-[#f2f3ff]"
                />
                <button
                  type="submit"
                  className="px-3 py-1 rounded-lg bg-[#006948] text-white text-xs font-semibold cursor-pointer"
                >
                  Save
                </button>
                <button
                  type="button"
                  onClick={() => setIsEditingGps(false)}
                  className="px-2 py-1 text-xs text-[#3d4a42] hover:text-[#131b2e] cursor-pointer"
                >
                  Cancel
                </button>
              </form>
            )}

            {gpsLocation.errorMessage && (
              <span className="text-[11px] text-[#ba1a1a] mt-0.5">
                {gpsLocation.errorMessage}
              </span>
            )}
          </div>
        </div>

        {/* GPS Action Buttons */}
        <div className="flex items-center gap-2 self-end md:self-center">
          <button
            type="button"
            onClick={() => acquireCurrentLocation(true)}
            disabled={gpsLocation.status === 'locating'}
            className="px-3 py-1.5 rounded-xl bg-[#eaedff] hover:bg-[#dae2fd] text-[#131b2e] text-xs font-semibold font-mono flex items-center gap-1.5 transition-colors cursor-pointer border border-[#dae2fd]"
            title="Acquire device GPS coordinates"
          >
            <span className={`material-symbols-outlined text-[16px] text-[#006948] ${gpsLocation.status === 'locating' ? 'animate-spin' : ''}`}>
              my_location
            </span>
            <span>Get Current GPS</span>
          </button>

          {!isEditingGps && (
            <button
              type="button"
              onClick={() => setIsEditingGps(true)}
              className="p-1.5 rounded-xl text-[#3d4a42] hover:bg-[#eaedff] transition-colors cursor-pointer"
              title="Edit Coordinates manually"
            >
              <span className="material-symbols-outlined text-[18px]">edit</span>
            </button>
          )}
        </div>
      </div>

      {/* Two Large Options (Camera prominent on mobile) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-stretch">
        {/* Mobile-first reordering: On mobile, camera is first! */}
        {/* Option 1: Use Camera */}
        <div className="order-1 md:order-2 bg-white p-7 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col justify-between gap-6 hover:shadow-md transition-shadow relative overflow-hidden">
          <div className="absolute top-0 right-0 px-3 py-1 bg-[#86f2e4]/30 text-[#006f66] font-mono text-[10px] font-bold rounded-bl-xl border-l border-b border-[#86f2e4]">
            IPHONE &bull; ANDROID READY
          </div>

          <div className="flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <span className="px-3 py-1 rounded-md bg-[#86f2e4]/30 text-[#006f66] font-mono text-xs font-bold">
                OPTION 01
              </span>
              <span className="material-symbols-outlined text-[#006a61] text-[24px]">
                photo_camera
              </span>
            </div>

            <div>
              <h2 className="text-xl font-bold text-[#131b2e]">Use Camera</h2>
              <p className="text-xs text-[#3d4a42] mt-1">
                Point your camera at a banana plant for instant field identification.
              </p>
            </div>

            {/* Viewfinder Preview Box */}
            <div
              onClick={() => setIsCameraOpen(true)}
              className="relative w-full h-48 rounded-xl overflow-hidden bg-[#dae2fd] border border-[#bccac0] cursor-pointer group flex items-center justify-center shadow-inner"
            >
              <img
                alt="Camera targeting preview"
                className="absolute inset-0 w-full h-full object-cover opacity-85 group-hover:scale-105 transition-transform duration-500"
                src={ASSETS.bananaFieldCalib}
              />
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-black/30" />

              {/* Pulsing Scanline */}
              <div className="absolute inset-x-0 h-0.5 bg-gradient-to-r from-transparent via-[#85f8c4] to-transparent shadow-[0_0_12px_#85f8c4] animate-[pulse_2s_infinite]" />

              <div className="relative z-10 flex flex-col items-center gap-1 text-white text-center px-4">
                <span className="material-symbols-outlined text-[32px] text-[#85f8c4] drop-shadow">
                  center_focus_strong
                </span>
                <span className="text-sm font-bold drop-shadow">Camera Viewfinder</span>
                <span className="font-mono text-[11px] text-[#f5fff7] bg-[#00855d]/85 px-3 py-0.5 rounded-full backdrop-blur-sm shadow-sm">
                  Tap to launch
                </span>
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-[#f2f3ff] border border-[#dae2fd] flex items-center gap-2.5 text-xs text-[#3d4a42]">
              <span className="material-symbols-outlined text-[#006948] text-[20px] shrink-0">
                info
              </span>
              <span>Point camera toward the banana tree pseudostem or full canopy foliage.</span>
            </div>
          </div>

          {/* Camera Button: Single robust Open Camera button */}
          <div className="pt-2">
            <button
              type="button"
              onClick={() => setIsCameraOpen(true)}
              className="w-full py-3.5 px-4 rounded-xl bg-[#006948] text-white hover:bg-[#00855d] font-semibold text-sm transition-all flex items-center justify-center gap-2 shadow-md shadow-[#006948]/25 cursor-pointer active:scale-[0.99]"
            >
              <span className="material-symbols-outlined text-[20px]">videocam</span>
              <span>Open Camera</span>
            </button>
          </div>
        </div>

        {/* Option 2: Upload Image */}
        <div className="order-2 md:order-1 bg-white p-7 rounded-2xl shadow-sm border border-[#dae2fd] flex flex-col justify-between gap-6 hover:shadow-md transition-shadow">
          <div className="flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <span className="px-3 py-1 rounded-md bg-[#eaedff] font-mono text-xs font-bold text-[#131b2e]">
                OPTION 02
              </span>
              <span className="material-symbols-outlined text-[#006948] text-[24px]">
                cloud_upload
              </span>
            </div>

            <div>
              <h2 className="text-xl font-bold text-[#131b2e]">Upload Image</h2>
              <p className="text-xs text-[#3d4a42] mt-1">
                Drag &amp; drop an image here or browse from your device.
              </p>
            </div>

            {/* Dropzone */}
            <label
              htmlFor="plant-image-upload"
              className="cursor-pointer flex flex-col items-center justify-center p-8 rounded-xl bg-[#f2f3ff] hover:bg-[#eaedff] border-2 border-dashed border-[#bccac0] hover:border-[#006948] transition-colors text-center relative group"
            >
              <div className="w-12 h-12 rounded-full bg-white flex items-center justify-center shadow-sm text-[#006948] mb-3 border border-[#dae2fd] group-hover:scale-105 transition-transform">
                <span className="material-symbols-outlined text-[26px]">add_photo_alternate</span>
              </div>
              <span className="text-sm font-bold text-[#131b2e]">Drag &amp; drop an image here</span>
              <span className="text-xs text-[#3d4a42] mt-1">
                or browse from your device &bull; Supports JPG, PNG, WEBP
              </span>
              <input
                ref={fileInputRef}
                id="plant-image-upload"
                type="file"
                accept="image/*"
                className="sr-only"
                onChange={handleFileUpload}
              />
            </label>

            {/* Benchmark Samples: 1 example to see */}
            <div className="flex flex-col gap-2 mt-1">
              <span className="font-mono text-xs text-[#3d4a42] font-semibold tracking-wide">
                QUICK BENCHMARK SAMPLE:
              </span>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => startAnalysis('positive')}
                  className="px-3 py-1.5 rounded-lg bg-[#eaedff] hover:bg-[#dae2fd] text-[#131b2e] font-mono text-xs flex items-center gap-1.5 transition-colors border border-[#dae2fd] cursor-pointer"
                >
                  <span className="material-symbols-outlined text-[14px] text-[#006948]">eco</span>
                  <span>Sample: Banana Tree Specimen</span>
                </button>
              </div>
            </div>
          </div>

          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="w-full py-3.5 px-4 rounded-xl bg-[#eaedff] text-[#131b2e] hover:bg-[#dae2fd] font-semibold text-sm transition-all flex items-center justify-center gap-2 border border-[#dae2fd] shadow-sm cursor-pointer"
          >
            <span className="material-symbols-outlined text-[18px]">folder_open</span>
            <span>Choose Image</span>
          </button>
        </div>
      </div>
    </div>
  );
};
