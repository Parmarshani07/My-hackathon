import { useEffect, useRef, useState, type ChangeEvent } from "react";

type SourceType = "webcam" | "upload";

type Detection = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
  classId: number;
  id?: string;
};

type DetectionLog = {
  id: number;
  objectName: string;
  time: string;
};

type CollisionEvent = {
  id: number;
  obj1: string;
  obj2: string;
  time: string;
};

type OrtTensor = {
  dims: number[];
  data: Float32Array;
};

type OrtSession = {
  inputNames: string[];
  outputNames: string[];
  run: (feeds: Record<string, unknown>) => Promise<Record<string, OrtTensor>>;
};

type OrtGlobal = {
  env: { wasm: { wasmPaths: string } };
  InferenceSession: {
    create: (
      modelUrl: string,
      options: { executionProviders: string[]; graphOptimizationLevel: string },
    ) => Promise<OrtSession>;
  };
  Tensor: new (type: string, data: Float32Array, dims: number[]) => unknown;
};

const MODEL_URLS = [
  "https://huggingface.co/unity/sentis-YOLOv8n/resolve/main/yolov8n.onnx",
  "https://huggingface.co/SpotLab/YOLOv8Detection/resolve/main/yolov8n.onnx",
  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx",
];

const INPUT_SIZE = 640;
const CONF_THRESHOLD = 0.35;
const IOU_THRESHOLD = 0.45;
const MAX_DETECTIONS = 50;
const COLLISION_ALERT_COOLDOWN_MS = 3000;
const ALERT_VISIBLE_MS = 2200;

// Color scheme for different object categories
const OBJECT_COLORS: Record<string, string> = {
  person: "#FFD700",           // Gold
  car: "#FF1493",              // Deep Pink
  truck: "#FF4500",            // Orange Red
  bus: "#FF0000",              // Red
  bicycle: "#00FF00",          // Lime
  motorcycle: "#FF6347",       // Tomato
  dog: "#FFB6C1",              // Light Pink
  cat: "#DDA0DD",              // Plum
  bird: "#87CEEB",             // Sky Blue
  default: "#00FFFF",          // Cyan
};

// Objects to track (expanded from just "person")
const DEFAULT_TRACKED_OBJECTS = new Set([
  "person", "car", "truck", "bus", "bicycle", "motorcycle",
  "dog", "cat", "bird", "boat", "traffic light", "stop sign",
]);

const COCO_CLASSES = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
  "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
  "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
  "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
  "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
];

function getOrtGlobal() {
  return (window as unknown as { ort?: OrtGlobal }).ort ?? null;
}

async function loadOrtScript() {
  if (getOrtGlobal()) return;

  await new Promise<void>((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js";
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("Failed to load ONNX Runtime script."));
    document.head.appendChild(script);
  });
}

function iou(a: Detection, b: Detection) {
  const interX1 = Math.max(a.x1, b.x1);
  const interY1 = Math.max(a.y1, b.y1);
  const interX2 = Math.min(a.x2, b.x2);
  const interY2 = Math.min(a.y2, b.y2);
  const interW = Math.max(0, interX2 - interX1);
  const interH = Math.max(0, interY2 - interY1);
  const interArea = interW * interH;
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  const union = areaA + areaB - interArea;
  return union <= 0 ? 0 : interArea / union;
}

// Collision detection: Check if two bounding boxes overlap
function checkCollision(a: Detection, b: Detection, overlapThreshold: number = 0.1): boolean {
  const overlapArea = iou(a, b);
  return overlapArea > overlapThreshold;
}

function applyNms(detections: Detection[]) {
  const sorted = [...detections].sort((a, b) => b.score - a.score);
  const picked: Detection[] = [];

  while (sorted.length > 0 && picked.length < MAX_DETECTIONS) {
    const current = sorted.shift() as Detection;
    picked.push(current);

    for (let i = sorted.length - 1; i >= 0; i -= 1) {
      const overlap = iou(current, sorted[i]);
      if (overlap > IOU_THRESHOLD && current.classId === sorted[i].classId) {
        sorted.splice(i, 1);
      }
    }
  }

  return picked;
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const processCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const uploadUrlRef = useRef<string | null>(null);
  const rafRef = useRef<number | null>(null);
  const sessionRef = useRef<OrtSession | null>(null);
  const runningInferenceRef = useRef(false);
  const alertTimeoutRef = useRef<number | null>(null);
  const lastAlertTimeRef = useRef<Map<string, number>>(new Map());
  const lastCollisionTimeRef = useRef<Map<string, number>>(new Map());
  const fpsCounterRef = useRef({ count: 0, fps: 0, lastTime: Date.now() });

  const [status, setStatus] = useState("Loading YOLO model...");
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<SourceType>("webcam");
  const [activeAlert, setActiveAlert] = useState<string | null>(null);
  const [detectionLogs, setDetectionLogs] = useState<DetectionLog[]>([]);
  const [collisionEvents, setCollisionEvents] = useState<CollisionEvent[]>([]);
  const [fps, setFps] = useState(0);
  const [objectCount, setObjectCount] = useState(0);
  const [collisionCount, setCollisionCount] = useState(0);
  const [trackedObjects, setTrackedObjects] = useState<Set<string>>(DEFAULT_TRACKED_OBJECTS);
  const [trackedObjectsList, setTrackedObjectsList] = useState<string[]>(Array.from(DEFAULT_TRACKED_OBJECTS));
  const [showStats, setShowStats] = useState(true);
  const [detectionThreshold, setDetectionThreshold] = useState(CONF_THRESHOLD);

  function stopFrameLoop() {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }

  function clearOverlay() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  function clearUploadedVideoUrl() {
    if (uploadUrlRef.current) {
      URL.revokeObjectURL(uploadUrlRef.current);
      uploadUrlRef.current = null;
    }
  }

  function clearAlertTimer() {
    if (alertTimeoutRef.current !== null) {
      window.clearTimeout(alertTimeoutRef.current);
      alertTimeoutRef.current = null;
    }
  }

  function triggerAlert(message: string, cooldownKey: string) {
    const now = Date.now();
    const lastSeenAt = lastAlertTimeRef.current.get(cooldownKey) ?? 0;
    if (now - lastSeenAt < COLLISION_ALERT_COOLDOWN_MS) return;

    lastAlertTimeRef.current.set(cooldownKey, now);
    setActiveAlert(message);

    clearAlertTimer();
    alertTimeoutRef.current = window.setTimeout(() => {
      setActiveAlert(null);
      alertTimeoutRef.current = null;
    }, ALERT_VISIBLE_MS);
  }

  function triggerDetectionAlert(objectName: string) {
    const time = new Date().toLocaleTimeString();
    setDetectionLogs((prev) => [{ id: Date.now(), objectName, time }, ...prev].slice(0, 12));
    triggerAlert(`🎯 Detected: ${objectName.toUpperCase()}`, `detection_${objectName}`);
  }

  function triggerCollisionAlert(obj1: string, obj2: string) {
    const key = [obj1, obj2].sort().join("_");
    const now = Date.now();
    const lastSeenAt = lastCollisionTimeRef.current.get(key) ?? 0;
    if (now - lastSeenAt < COLLISION_ALERT_COOLDOWN_MS) return;

    lastCollisionTimeRef.current.set(key, now);
    const time = new Date().toLocaleTimeString();
    
    setCollisionEvents((prev) => [{ id: now, obj1, obj2, time }, ...prev].slice(0, 10));
    setCollisionCount((prev) => prev + 1);
    triggerAlert(`⚠️ COLLISION: ${obj1.toUpperCase()} ↔ ${obj2.toUpperCase()}`, key);
  }

  function getObjectColor(classId: number): string {
    const label = COCO_CLASSES[classId] ?? "default";
    return OBJECT_COLORS[label] || OBJECT_COLORS["default"];
  }

  function shouldTrackObject(label: string): boolean {
    return trackedObjects.has(label);
  }

  function syncCanvasToVideo() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.videoWidth === 0 || video.videoHeight === 0) return;
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
  }

  async function loadModel() {
    await loadOrtScript();

    const ort = getOrtGlobal();
    if (!ort) throw new Error("ONNX Runtime is not available.");

    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

    let lastError = "";
    for (const modelUrl of MODEL_URLS) {
      try {
        const session = await ort.InferenceSession.create(modelUrl, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });
        sessionRef.current = session;
        return;
      } catch (e) {
        lastError = e instanceof Error ? e.message : "Failed to load model.";
      }
    }

    throw new Error(lastError || "Unable to load YOLO model from known URLs.");
  }

  function preprocessFrame(video: HTMLVideoElement) {
    const ort = getOrtGlobal();
    if (!ort) {
      throw new Error("ONNX Runtime is not loaded.");
    }

    if (!processCanvasRef.current) {
      processCanvasRef.current = document.createElement("canvas");
    }

    const canvas = processCanvasRef.current;
    canvas.width = INPUT_SIZE;
    canvas.height = INPUT_SIZE;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) {
      throw new Error("Could not get canvas context for preprocessing.");
    }

    ctx.drawImage(video, 0, 0, INPUT_SIZE, INPUT_SIZE);
    const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
    const pixels = INPUT_SIZE * INPUT_SIZE;
    const data = new Float32Array(3 * pixels);

    for (let i = 0; i < pixels; i += 1) {
      const base = i * 4;
      data[i] = imageData[base] / 255;
      data[pixels + i] = imageData[base + 1] / 255;
      data[2 * pixels + i] = imageData[base + 2] / 255;
    }

    return new ort.Tensor("float32", data, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  }

  function decodeOutput(output: OrtTensor, videoWidth: number, videoHeight: number) {
    const dims = output.dims;
    const data = output.data as Float32Array;

    let numPreds = 0;
    let channels = 0;
    let channelFirst = true;

    if (dims.length === 3 && dims[1] > dims[2]) {
      channels = dims[1];
      numPreds = dims[2];
      channelFirst = true;
    } else if (dims.length === 3) {
      numPreds = dims[1];
      channels = dims[2];
      channelFirst = false;
    } else {
      throw new Error(`Unexpected YOLO output shape: ${dims.join("x")}`);
    }

    const detections: Detection[] = [];
    const classCount = channels - 4;

    for (let i = 0; i < numPreds; i += 1) {
      const read = (offset: number) => {
        if (channelFirst) return data[offset * numPreds + i];
        return data[i * channels + offset];
      };

      const cx = read(0);
      const cy = read(1);
      const w = read(2);
      const h = read(3);

      let classId = -1;
      let classScore = 0;
      for (let c = 0; c < classCount; c += 1) {
        const score = read(4 + c);
        if (score > classScore) {
          classScore = score;
          classId = c;
        }
      }

      if (classScore < detectionThreshold || classId < 0) continue;

      const x1 = ((cx - w / 2) / INPUT_SIZE) * videoWidth;
      const y1 = ((cy - h / 2) / INPUT_SIZE) * videoHeight;
      const x2 = ((cx + w / 2) / INPUT_SIZE) * videoWidth;
      const y2 = ((cy + h / 2) / INPUT_SIZE) * videoHeight;

      detections.push({
        x1: Math.max(0, x1),
        y1: Math.max(0, y1),
        x2: Math.min(videoWidth, x2),
        y2: Math.min(videoHeight, y2),
        score: classScore,
        classId,
        id: `${classId}_${i}_${Date.now()}`,
      });
    }

    return applyNms(detections);
  }

  function drawDetections(detections: Detection[]) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw all detections first
    for (const det of detections) {
      const label = COCO_CLASSES[det.classId] ?? `class ${det.classId}`;
      const text = `${label} ${(det.score * 100).toFixed(0)}%`;
      const color = getObjectColor(det.classId);
      const width = det.x2 - det.x1;
      const height = det.y2 - det.y1;

      // Draw box with glow effect if tracked
      if (shouldTrackObject(label)) {
        ctx.shadowColor = color;
        ctx.shadowBlur = 10;
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 0;
      }

      ctx.lineWidth = 3;
      ctx.strokeStyle = color;
      ctx.strokeRect(det.x1, det.y1, width, height);

      // Draw label background
      ctx.font = "bold 14px sans-serif";
      const textMetrics = ctx.measureText(text);
      const textWidth = textMetrics.width;
      const textX = det.x1;
      const textY = Math.max(0, det.y1 - 24);

      ctx.fillStyle = color;
      ctx.fillRect(textX - 2, textY - 2, textWidth + 8, 24);

      // Draw text
      ctx.fillStyle = shouldTrackObject(label) ? "#000000" : "#FFFFFF";
      ctx.fillText(text, textX + 2, textY + 16);

      ctx.shadowBlur = 0;
    }
  }

  function detectCollisions(detections: Detection[]) {
    for (let i = 0; i < detections.length; i++) {
      for (let j = i + 1; j < detections.length; j++) {
        if (checkCollision(detections[i], detections[j], 0.05)) {
          const label1 = COCO_CLASSES[detections[i].classId] ?? "unknown";
          const label2 = COCO_CLASSES[detections[j].classId] ?? "unknown";
          triggerCollisionAlert(label1, label2);
        }
      }
    }
  }

  async function processFrame() {
    const video = videoRef.current;
    const session = sessionRef.current;
    if (!video || !session) return;
    if (video.paused || video.ended || video.readyState < 2) return;
    if (runningInferenceRef.current) return;

    runningInferenceRef.current = true;
    try {
      syncCanvasToVideo();
      const input = preprocessFrame(video);
      const inputName = session.inputNames[0];
      const outputMap = await session.run({ [inputName]: input });
      const outputName = session.outputNames[0];
      const output = outputMap[outputName];
      if (!output) {
        throw new Error("Model did not return an output tensor.");
      }

      const detections = decodeOutput(output, video.videoWidth, video.videoHeight);
      
      // Filter for tracked objects
      const trackedDetections = detections.filter((det) => {
        const label = COCO_CLASSES[det.classId] ?? "unknown";
        return shouldTrackObject(label);
      });

      drawDetections(detections);
      
      // Check for collisions among tracked objects
      if (trackedDetections.length > 1) {
        detectCollisions(trackedDetections);
      }

      // Log detections
      for (const det of trackedDetections) {
        const label = COCO_CLASSES[det.classId] ?? "unknown";
        triggerDetectionAlert(label);
      }

      setObjectCount(trackedDetections.length);

      // Update FPS
      const now = Date.now();
      fpsCounterRef.current.count++;
      if (now - fpsCounterRef.current.lastTime >= 1000) {
        fpsCounterRef.current.fps = fpsCounterRef.current.count;
        fpsCounterRef.current.count = 0;
        fpsCounterRef.current.lastTime = now;
        setFps(fpsCounterRef.current.fps);
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : "Frame processing failed.";
      setError(message);
    } finally {
      runningInferenceRef.current = false;
    }
  }

  function startFrameLoop() {
    stopFrameLoop();
    const loop = async () => {
      await processFrame();
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
  }

  async function startWebcam() {
    setError(null);
    setSource("webcam");
    clearUploadedVideoUrl();
    setObjectCount(0);
    setCollisionCount(0);
    setCollisionEvents([]);
    setDetectionLogs([]);

    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus("Webcam API not supported in this browser.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });

      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) return;

      video.srcObject = stream;
      video.removeAttribute("src");
      video.controls = false;
      await video.play();

      setStatus(`🔴 Live Webcam - Tracking: ${trackedObjectsList.join(", ")}`);
      startFrameLoop();
    } catch (e) {
      const message = e instanceof Error ? e.message : "Could not open webcam.";
      setStatus("Webcam unavailable");
      setError(message);
    }
  }

  async function handleVideoUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    stopWebcam();
    setSource("upload");
    setError(null);
    clearUploadedVideoUrl();
    setObjectCount(0);
    setCollisionCount(0);
    setCollisionEvents([]);
    setDetectionLogs([]);

    const video = videoRef.current;
    if (!video) return;

    const url = URL.createObjectURL(file);
    uploadUrlRef.current = url;
    video.srcObject = null;
    video.src = url;
    video.controls = true;

    try {
      await video.play();
      setStatus(`📹 Playing: ${file.name}`);
    } catch {
      setStatus(`📹 Uploaded: ${file.name}`);
      setError("Autoplay may be blocked. Press play on the video player.");
    }

    startFrameLoop();
  }

  function stopWebcam() {
    stopFrameLoop();

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    const video = videoRef.current;
    if (video) {
      video.pause();
      video.srcObject = null;
    }

    clearOverlay();
    clearAlertTimer();
    setActiveAlert(null);
    setStatus("⏸️ Stopped");
  }

  function toggleObjectTracking(objectName: string) {
    const newTracked = new Set(trackedObjects);
    if (newTracked.has(objectName)) {
      newTracked.delete(objectName);
    } else {
      newTracked.add(objectName);
    }
    setTrackedObjects(newTracked);
    setTrackedObjectsList(Array.from(newTracked).sort());
  }

  function trackAllObjects() {
    const allClasses = new Set(COCO_CLASSES.filter((c) => c && c !== "unknown"));
    setTrackedObjects(allClasses);
    setTrackedObjectsList(Array.from(allClasses).sort());
  }

  function clearTracking() {
    setTrackedObjects(new Set());
    setTrackedObjectsList([]);
  }

  useEffect(() => {
    let active = true;

    const setup = async () => {
      try {
        await loadModel();
        if (!active) return;
        setStatus("✅ Model loaded. Starting webcam...");
        await startWebcam();
      } catch (e) {
        if (!active) return;
        const message = e instanceof Error ? e.message : "Could not initialize detector.";
        setStatus("❌ Failed to start detector");
        setError(message);
      }
    };

    void setup();

    return () => {
      active = false;
      stopWebcam();
      clearUploadedVideoUrl();
      clearAlertTimer();
    };
  }, []);

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 px-4 py-8 text-slate-100">
      <div className="mx-auto w-full max-w-6xl space-y-4">
        <div className="flex items-end justify-between">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
              🎯 YOLO Pro Detector
            </h1>
            <p className="text-sm text-slate-400 mt-1">Advanced object detection with collision alerts</p>
          </div>
          <div className="text-right">
            <p className="text-lg font-bold text-cyan-400">{fps} FPS</p>
            <p className="text-sm text-slate-400">Objects: {objectCount}</p>
          </div>
        </div>

        {activeAlert ? (
          <div className="rounded-lg border-2 border-yellow-300 bg-gradient-to-r from-yellow-500 to-red-500 px-4 py-3 text-base font-bold text-white shadow-2xl animate-pulse">
            {activeAlert}
          </div>
        ) : null}

        <div className="relative overflow-hidden rounded-lg border-2 border-cyan-500 bg-black shadow-2xl">
          <video ref={videoRef} playsInline muted autoPlay className="h-auto w-full" />
          <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2 space-y-4">
            <div className="bg-slate-800 border border-slate-700 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-cyan-400 mb-3">Controls</h3>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={startWebcam}
                  className="rounded-md bg-green-500 px-4 py-2 text-sm font-medium text-white hover:bg-green-600 active:scale-95 transition"
                >
                  🔴 Start Webcam
                </button>
                <button
                  type="button"
                  onClick={stopWebcam}
                  className="rounded-md bg-red-500 px-4 py-2 text-sm font-medium text-white hover:bg-red-600 active:scale-95 transition"
                >
                  ⏸️ Stop
                </button>
                <label className="cursor-pointer rounded-md bg-blue-500 px-4 py-2 text-sm font-medium text-white hover:bg-blue-600 active:scale-95 transition">
                  📹 Upload Video
                  <input type="file" accept="video/*" onChange={handleVideoUpload} className="hidden" />
                </label>
              </div>
              <p className="text-xs text-slate-400 mt-3">Status: {status}</p>
              {error ? <p className="text-xs text-red-400 mt-1">⚠️ {error}</p> : null}
            </div>

            <div className="bg-slate-800 border border-slate-700 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-cyan-400 mb-3">Detection Threshold</h3>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={detectionThreshold}
                onChange={(e) => setDetectionThreshold(parseFloat(e.target.value))}
                className="w-full"
              />
              <p className="text-xs text-slate-400 mt-2">Confidence: {(detectionThreshold * 100).toFixed(0)}%</p>
            </div>
          </div>

          <div className="space-y-4">
            <div className="bg-slate-800 border border-slate-700 rounded-lg p-4">
              <div className="flex justify-between items-center mb-3">
                <h3 className="text-sm font-semibold text-cyan-400">Stats</h3>
                <button
                  type="button"
                  onClick={() => setShowStats(!showStats)}
                  className="text-xs px-2 py-1 bg-slate-700 hover:bg-slate-600 rounded"
                >
                  {showStats ? "Hide" : "Show"}
                </button>
              </div>
              {showStats && (
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Tracked Objects:</span>
                    <span className="font-bold text-yellow-400">{objectCount}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Collisions:</span>
                    <span className="font-bold text-red-400">{collisionCount}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">FPS:</span>
                    <span className="font-bold text-green-400">{fps}</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="bg-slate-800 border border-slate-700 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-cyan-400 mb-3">🎯 Track Objects</h3>
          <div className="space-y-3">
            <div className="flex gap-2">
              <button
                type="button"
                onClick={trackAllObjects}
                className="rounded-md bg-indigo-600 px-3 py-1 text-xs font-medium text-white hover:bg-indigo-700"
              >
                Track All
              </button>
              <button
                type="button"
                onClick={() => setTrackedObjects(DEFAULT_TRACKED_OBJECTS) || setTrackedObjectsList(Array.from(DEFAULT_TRACKED_OBJECTS))}
                className="rounded-md bg-blue-600 px-3 py-1 text-xs font-medium text-white hover:bg-blue-700"
              >
                Reset Default
              </button>
              <button
                type="button"
                onClick={clearTracking}
                className="rounded-md bg-slate-600 px-3 py-1 text-xs font-medium text-white hover:bg-slate-700"
              >
                Clear All
              </button>
            </div>
            <div className="flex flex-wrap gap-2">
              {["person", "car", "truck", "bus", "bicycle", "dog", "cat", "bird"].map((obj) => (
                <button
                  key={obj}
                  type="button"
                  onClick={() => toggleObjectTracking(obj)}
                  className={`rounded-md px-3 py-1 text-xs font-medium transition ${
                    trackedObjects.has(obj)
                      ? "bg-green-600 text-white hover:bg-green-700"
                      : "bg-slate-700 text-slate-300 hover:bg-slate-600"
                  }`}
                >
                  {obj}
                </button>
              ))}
            </div>
            {trackedObjectsList.length > 0 && (
              <p className="text-xs text-slate-400">Tracking: {trackedObjectsList.join(", ")}</p>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-cyan-400 mb-3">📊 Detection Log</h3>
            {detectionLogs.length === 0 ? (
              <p className="text-xs text-slate-400">No detections yet.</p>
            ) : (
              <ul className="space-y-1 max-h-48 overflow-y-auto">
                {detectionLogs.map((entry) => (
                  <li key={entry.id} className="text-xs text-slate-300">
                    <span className="text-slate-500">[{entry.time}]</span> ✓ {entry.objectName}
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="bg-slate-800 border border-slate-700 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-red-400 mb-3">⚠️ Collision Log</h3>
            {collisionEvents.length === 0 ? (
              <p className="text-xs text-slate-400">No collisions detected.</p>
            ) : (
              <ul className="space-y-1 max-h-48 overflow-y-auto">
                {collisionEvents.map((entry) => (
                  <li key={entry.id} className="text-xs text-red-300">
                    <span className="text-slate-500">[{entry.time}]</span> ⚠️ {entry.obj1} ↔ {entry.obj2}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}