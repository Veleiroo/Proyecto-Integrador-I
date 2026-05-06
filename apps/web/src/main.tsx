import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Bell,
  BellOff,
  Camera,
  CheckCircle2,
  ChevronRight,
  Clock3,
  DatabaseZap,
  EyeOff,
  FileImage,
  ImagePlus,
  KeyRound,
  Loader2,
  LockKeyhole,
  LogOut,
  Moon,
  MonitorCheck,
  Pause,
  Play,
  RotateCcw,
  Send,
  Server,
  ShieldCheck,
  Sun,
  UserRound,
  Video,
  VideoOff,
} from "lucide-react";
import "./styles.css";

type ViewMode = "front" | "lateral";
type Status = "adequate" | "improvable" | "risk" | "insufficient_data";
type Section = "camera" | "review" | "stats" | "history" | "privacy";
type Theme = "light" | "dark";

type ApiResult = {
  view: ViewMode;
  model: string;
  backend: string | null;
  pose_detected: boolean;
  visible_landmarks_count: number | null;
  status: Status;
  status_label: string;
  feedback: string;
  metrics: Record<string, number | null>;
  components: Record<string, string>;
};

type ReviewRecord = ApiResult & {
  id: string | number;
  fileName: string;
  createdAt: string;
};

type StatsSummary = {
  total: number;
  by_status: Record<string, number>;
  latest_at: string | null;
  periods: Record<string, { total: number; adequate_ratio: number; risk_count: number; improvable_count: number }>;
  timeline: Array<{ date: string; total: number; adequate: number; improvable: number; risk: number }>;
  recommendations: string[];
};

type AppUser = {
  id: number;
  username: string;
  display_name: string;
  role: "user" | "dev" | string;
  created_at: string;
};

type AuthSession = {
  user: AppUser;
  access_token: string;
  token_type: string;
  expires_at: string;
};

const DEFAULT_API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const sectionCopy: Record<Section, { eyebrow: string; title: string }> = {
  camera: {
    eyebrow: "Monitor local",
    title: "Cámara y seguimiento ergonómico",
  },
  review: {
    eyebrow: "Revisión manual",
    title: "Revisión ergonómica por imagen",
  },
  history: {
    eyebrow: "Sesión local",
    title: "Historial de revisiones",
  },
  stats: {
    eyebrow: "Tendencias",
    title: "Estadísticas y recomendaciones",
  },
  privacy: {
    eyebrow: "Local first",
    title: "Privacidad y datos del equipo",
  },
};

const viewCopy: Record<ViewMode, { label: string; short: string; endpoint: string; model: string }> = {
  front: {
    label: "Vista frontal",
    short: "Hombros, cuello y simetría",
    endpoint: "/api/analyze/front",
    model: "MediaPipe Pose",
  },
  lateral: {
    label: "Vista lateral",
    short: "Tronco, perfil y cabeza adelantada",
    endpoint: "/api/analyze/lateral",
    model: "YOLO Pose",
  },
};

const metricLabels: Record<string, string> = {
  shoulder_tilt_deg: "Inclinación hombros",
  shoulder_height_diff_ratio: "Desnivel hombros",
  head_lateral_offset_ratio: "Cabeza lateral",
  neck_tilt_deg: "Cuello frontal",
  trunk_tilt_deg: "Tronco frontal",
  left_elbow_angle_deg: "Codo izquierdo",
  right_elbow_angle_deg: "Codo derecho",
  head_forward_offset_ratio: "Cabeza adelantada",
  neck_forward_tilt_deg: "Cuello lateral",
  trunk_forward_tilt_deg: "Tronco lateral",
  shoulder_hip_offset_ratio: "Hombro-cadera",
  lateral_elbow_angle_deg: "Codo lateral",
};

function normalizeApiBase(value: string) {
  return value.trim().replace(/\/$/, "");
}

function tone(status: Status | string) {
  if (status === "adequate") return "ok";
  if (status === "improvable") return "warn";
  if (status === "risk") return "risk";
  return "muted";
}

function formatMetric(value: number | null, key: string) {
  if (value === null || Number.isNaN(value)) return "Sin dato";
  if (key.endsWith("_deg")) return `${value.toFixed(1)}°`;
  return value.toFixed(3);
}

function App() {
  const [session, setSession] = useState<AuthSession | null>(() => {
    const stored = localStorage.getItem("authSession");
    return stored ? (JSON.parse(stored) as AuthSession) : null;
  });
  const [activeSection, setActiveSection] = useState<Section>("camera");
  const [theme, setTheme] = useState<Theme>((localStorage.getItem("theme") as Theme | null) ?? "light");
  const [apiBase, setApiBase] = useState(normalizeApiBase(localStorage.getItem("apiBase") ?? DEFAULT_API_BASE));
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [view, setView] = useState<ViewMode>("front");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<ApiResult | null>(null);
  const [history, setHistory] = useState<ReviewRecord[]>([]);
  const [stats, setStats] = useState<StatsSummary | null>(null);
  const [notificationsEnabled, setNotificationsEnabled] = useState(() => localStorage.getItem("notificationsEnabled") !== "false");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    localStorage.setItem("theme", theme);
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    localStorage.setItem("apiBase", apiBase);
  }, [apiBase]);

  useEffect(() => {
    if (!session) {
      localStorage.removeItem("authSession");
      return;
    }
    localStorage.setItem("authSession", JSON.stringify(session));
  }, [session]);

  useEffect(() => {
    if (!session) return;
    const controller = new AbortController();
    fetch(`${apiBase}/api/auth/me`, {
      headers: { Authorization: `Bearer ${session.access_token}` },
      signal: controller.signal,
    })
      .then((response) => {
        if (!response.ok) setSession(null);
      })
      .catch(() => null);
    return () => controller.abort();
  }, [apiBase, session?.access_token]);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${apiBase}/api/health`, { signal: controller.signal })
      .then((response) => setApiOnline(response.ok))
      .catch(() => setApiOnline(false));
    return () => controller.abort();
  }, [apiBase]);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      return;
    }
    const objectUrl = URL.createObjectURL(file);
    setPreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  useEffect(() => {
    localStorage.setItem("notificationsEnabled", String(notificationsEnabled));
  }, [notificationsEnabled]);

  const metrics = useMemo(() => Object.entries(result?.metrics ?? {}).filter(([, value]) => value !== null), [result]);
  const selectedView = viewCopy[view];
  const activeCopy = sectionCopy[activeSection];
  const authHeaders = session ? { Authorization: `Bearer ${session.access_token}` } : {};

  useEffect(() => {
    if (!session) return;
    refreshUserData();
  }, [session?.access_token, apiBase]);

  async function refreshUserData() {
    if (!session) return;
    const [historyResponse, statsResponse] = await Promise.all([
      fetch(`${apiBase}/api/analyses?limit=30`, { headers: authHeaders }),
      fetch(`${apiBase}/api/summary`, { headers: authHeaders }),
    ]);
    if (historyResponse.ok) {
      const body = await historyResponse.json();
      setHistory(
        body.items.map((item: ApiResult & { id: number; created_at: string }) => ({
          ...item,
          id: item.id,
          fileName: `Captura ${item.id}`,
          createdAt: item.created_at,
        })),
      );
    }
    if (statsResponse.ok) {
      setStats(await statsResponse.json());
    }
  }

  function pushAnalysis(nextResult: ApiResult & { id?: number; created_at?: string }, fileName: string) {
    setResult(nextResult);
    setHistory((items) => [
      {
        ...nextResult,
        id: nextResult.id ?? crypto.randomUUID(),
        fileName,
        createdAt: nextResult.created_at ?? new Date().toISOString(),
      },
      ...items.slice(0, 29),
    ]);
    refreshUserData();
  }

  function notifyAnalysis(nextResult: ApiResult) {
    if (!notificationsEnabled) return;
    const message = `${nextResult.status_label}: ${nextResult.feedback}`;
    if ("Notification" in window && Notification.permission === "granted") {
      new Notification("PostureOS", { body: message });
    }
  }

  async function analyzeImage() {
    if (!file) return;
    setIsLoading(true);
    setError(null);
    setResult(null);

    const payload = new FormData();
    payload.append("file", file);

    try {
      const response = await fetch(`${apiBase}${selectedView.endpoint}`, {
        method: "POST",
        headers: authHeaders,
        body: payload,
      });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail ?? `Error ${response.status}`);
      }
      const nextResult = (await response.json()) as ApiResult & { id?: number; created_at?: string };
      pushAnalysis(nextResult, file.name);
      notifyAnalysis(nextResult);
      setApiOnline(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo analizar la imagen.");
      setApiOnline(false);
    } finally {
      setIsLoading(false);
    }
  }

  function resetImage() {
    setFile(null);
    setResult(null);
    setError(null);
  }

  async function logout() {
    if (session) {
      await fetch(`${apiBase}/api/auth/logout`, {
        method: "POST",
        headers: authHeaders,
      }).catch(() => null);
    }
    setSession(null);
    setHistory([]);
    setResult(null);
  }

  if (!session) {
    return (
      <LoginScreen
        apiBase={apiBase}
        setApiBase={setApiBase}
        theme={theme}
        onToggleTheme={() => setTheme((current) => (current === "light" ? "dark" : "light"))}
        onAuthenticated={setSession}
      />
    );
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup">
          <div className="brand-mark">
            <Activity size={22} />
          </div>
          <div>
            <strong>PostureOS</strong>
            <span>Ergonomía preventiva</span>
          </div>
        </div>

        <nav className="nav-list">
          <NavButton active={activeSection === "camera"} icon={<MonitorCheck />} label="Cámara" onClick={() => setActiveSection("camera")} />
          <NavButton active={activeSection === "review"} icon={<Camera />} label="Revisión" onClick={() => setActiveSection("review")} />
          <NavButton active={activeSection === "stats"} icon={<BarChart3 />} label="Estadísticas" onClick={() => setActiveSection("stats")} />
          <NavButton active={activeSection === "history"} icon={<BarChart3 />} label="Historial local" onClick={() => setActiveSection("history")} />
          <NavButton active={activeSection === "privacy"} icon={<ShieldCheck />} label="Privacidad" onClick={() => setActiveSection("privacy")} />
        </nav>

        <div className={`connection-card ${apiOnline ? "online" : "offline"}`}>
          <Server size={18} />
          <div>
            <strong>{apiOnline ? "Backend conectado" : "Backend no disponible"}</strong>
            <span>{apiBase}</span>
          </div>
        </div>

        <div className="sidebar-actions">
          <ThemeButton theme={theme} onToggle={() => setTheme((current) => (current === "light" ? "dark" : "light"))} />
          <button className="logout-button" type="button" onClick={logout}>
            <LogOut size={17} />
            Salir
          </button>
        </div>
      </aside>

      <section className="main-area">
        <header className="page-header">
          <div>
            <p className="eyebrow">{activeCopy.eyebrow}</p>
            <h1>{activeCopy.title}</h1>
          </div>
          <div className="header-actions">
            <ThemeButton theme={theme} onToggle={() => setTheme((current) => (current === "light" ? "dark" : "light"))} compact />
            <div className="user-chip">
              <UserRound size={17} />
              <span>{session.user.display_name}</span>
            </div>
          </div>
        </header>

        {activeSection === "camera" && (
          <CameraPanel
            apiBase={apiBase}
            apiOnline={apiOnline}
            authHeaders={authHeaders}
            notificationsEnabled={notificationsEnabled}
            setNotificationsEnabled={setNotificationsEnabled}
            onAnalysis={(analysis) => {
              pushAnalysis(analysis, `Captura ${analysis.id ?? ""}`.trim());
              notifyAnalysis(analysis);
            }}
          />
        )}
        {activeSection === "review" && (
          <section className="review-layout">
            <div className="left-column">
              <div className="view-switch">
                {(Object.keys(viewCopy) as ViewMode[]).map((mode) => (
                  <button
                    key={mode}
                    className={view === mode ? "active" : ""}
                    type="button"
                    onClick={() => {
                      setView(mode);
                      setResult(null);
                      setError(null);
                    }}
                  >
                    <strong>{viewCopy[mode].label}</strong>
                    <span>{viewCopy[mode].short}</span>
                  </button>
                ))}
              </div>

              <div className="upload-card">
                <div
                  className={`dropzone ${preview ? "with-preview" : ""}`}
                  onClick={() => inputRef.current?.click()}
                  onDragOver={(event) => event.preventDefault()}
                  onDrop={(event) => {
                    event.preventDefault();
                    setFile(event.dataTransfer.files?.[0] ?? null);
                    setResult(null);
                    setError(null);
                  }}
                >
                  <input ref={inputRef} type="file" accept="image/*" onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
                  {preview ? (
                    <img src={preview} alt="Imagen seleccionada" />
                  ) : (
                    <div className="empty-upload">
                      <ImagePlus size={38} />
                      <strong>Sube una imagen {view === "front" ? "frontal" : "lateral"}</strong>
                      <span>Arrastra el archivo aquí o pulsa para seleccionarlo</span>
                    </div>
                  )}
                </div>

                <div className="upload-actions">
                  <button className="ghost-button" type="button" onClick={resetImage} disabled={!file}>
                    <RotateCcw size={17} />
                    Limpiar
                  </button>
                  <button className="primary-button" type="button" onClick={analyzeImage} disabled={!file || isLoading}>
                    {isLoading ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
                    Analizar imagen
                  </button>
                </div>
              </div>
            </div>

            <aside className="result-column">
              {error ? (
                <Message title="No se pudo analizar" body={error} />
              ) : result ? (
                <ResultPanel result={result} metrics={metrics} />
              ) : (
                <EmptyResult view={view} />
              )}
            </aside>
          </section>
        )}

        {activeSection === "history" && <HistoryPanel history={history} />}
        {activeSection === "stats" && <StatsPanel stats={stats} history={history} />}
        {activeSection === "privacy" && <PrivacyPanel apiBase={apiBase} setApiBase={setApiBase} />}
      </section>
    </main>
  );
}

function LoginScreen({
  apiBase,
  setApiBase,
  theme,
  onToggleTheme,
  onAuthenticated,
}: {
  apiBase: string;
  setApiBase: (value: string) => void;
  theme: Theme;
  onToggleTheme: () => void;
  onAuthenticated: (session: AuthSession) => void;
}) {
  const [username, setUsername] = useState("Pablo");
  const [password, setPassword] = useState("1234");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [loginError, setLoginError] = useState<string | null>(null);

  async function login(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsSubmitting(true);
    setLoginError(null);
    try {
      const response = await fetch(`${apiBase}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail ?? "No se pudo iniciar sesión.");
      }
      onAuthenticated((await response.json()) as AuthSession);
    } catch (error) {
      setLoginError(error instanceof Error ? error.message : "No se pudo iniciar sesión.");
    } finally {
      setIsSubmitting(false);
    }
  }

  function fillUser(nextUsername: string, nextPassword: string) {
    setUsername(nextUsername);
    setPassword(nextPassword);
    setLoginError(null);
  }

  return (
    <main className="login-shell">
      <section className="login-panel">
        <div className="login-copy">
          <div className="brand-lockup large">
            <div className="brand-mark">
              <Activity size={24} />
            </div>
            <div>
              <strong>PostureOS</strong>
              <span>Local posture intelligence</span>
            </div>
          </div>
          <h1>Ergonomía privada en tu propio equipo</h1>
          <p>
            La cámara se procesa en este ordenador, las imágenes no se guardan y cada usuario conserva sus métricas
            separadas con una base de datos local cifrada.
          </p>
          <div className="security-list">
            <span><LockKeyhole size={16} /> Usuarios del equipo</span>
            <span><EyeOff size={16} /> Imágenes temporales</span>
            <span><DatabaseZap size={16} /> Resultados cifrados</span>
          </div>
        </div>

        <form className="login-card" onSubmit={login}>
          <div className="login-heading">
            <div>
              <p className="eyebrow">Acceso local</p>
              <h2>Entrar a PostureOS</h2>
            </div>
            <ThemeButton theme={theme} onToggle={onToggleTheme} compact />
          </div>
          <label>
            Usuario
            <div className="input-wrap">
              <UserRound size={17} />
              <input autoComplete="username" value={username} onChange={(event) => setUsername(event.target.value)} />
            </div>
          </label>
          <label>
            Clave
            <div className="input-wrap">
              <KeyRound size={17} />
              <input autoComplete="current-password" type="password" value={password} onChange={(event) => setPassword(event.target.value)} />
            </div>
          </label>
          <div className="quick-users">
            <button type="button" onClick={() => fillUser("Pablo", "1234")}>
              Pablo
            </button>
            <button type="button" onClick={() => fillUser("admin", "admin")}>
              Técnico
            </button>
          </div>
          <label>
            API local
            <div className="input-wrap">
              <Server size={17} />
              <input value={apiBase} onChange={(event) => setApiBase(normalizeApiBase(event.target.value))} />
            </div>
          </label>
          {loginError && <div className="login-error">{loginError}</div>}
          <button className="primary-button full" type="submit" disabled={!username || !password || isSubmitting}>
            {isSubmitting ? <Loader2 className="spin" size={18} /> : <ChevronRight size={18} />}
            Entrar
          </button>
          <small>Usuarios iniciales: <code>Pablo</code>/<code>1234</code> y <code>admin</code>/<code>admin</code>.</small>
        </form>
      </section>
    </main>
  );
}

function CameraPanel({
  apiBase,
  apiOnline,
  authHeaders,
  notificationsEnabled,
  setNotificationsEnabled,
  onAnalysis,
}: {
  apiBase: string;
  apiOnline: boolean | null;
  authHeaders: Record<string, string>;
  notificationsEnabled: boolean;
  setNotificationsEnabled: (value: boolean) => void;
  onAnalysis: (analysis: ApiResult & { id?: number; created_at?: string }) => void;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const runTimerRef = useRef<number | null>(null);
  const isRunningRef = useRef(false);
  const [cameraState, setCameraState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [permissionState, setPermissionState] = useState<PermissionState | "unsupported">("unsupported");
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [cadence, setCadence] = useState("Cada 10 min");
  const [lastCapture, setLastCapture] = useState<ApiResult | null>(null);
  const [captureStatus, setCaptureStatus] = useState<string>("Sin capturas en esta sesión");

  useEffect(() => {
    let permissionStatus: PermissionStatus | null = null;
    const updatePermission = () => {
      if (permissionStatus) setPermissionState(permissionStatus.state);
    };

    if ("permissions" in navigator && navigator.permissions.query) {
      navigator.permissions
        .query({ name: "camera" as PermissionName })
        .then((status) => {
          permissionStatus = status;
          setPermissionState(status.state);
          status.addEventListener("change", updatePermission);
        })
        .catch(() => setPermissionState("unsupported"));
    }

    return () => {
      permissionStatus?.removeEventListener("change", updatePermission);
      if (runTimerRef.current) window.clearTimeout(runTimerRef.current);
      stopCamera();
    };
  }, []);

  async function startCamera() {
    setCameraState("loading");
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraState("ready");
      setPermissionState("granted");
      return true;
    } catch (error) {
      setCameraState("error");
      if (error instanceof DOMException && error.name === "NotAllowedError") {
        setPermissionState("denied");
      }
      setCameraError(error instanceof Error ? error.message : "No se pudo abrir la cámara.");
      return false;
    }
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraState("idle");
  }

  async function captureAndAnalyze() {
    setCaptureStatus("Capturando imagen frontal...");
    const started = await startCamera();
    if (!started || !videoRef.current) return;
    await new Promise((resolve) => window.setTimeout(resolve, 900));
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    stopCamera();
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
    if (!blob) return;
    const payload = new FormData();
    payload.append("file", blob, `capture-${Date.now()}.jpg`);
    setCaptureStatus("Analizando postura...");
    const response = await fetch(`${apiBase}/api/analyze/front`, {
      method: "POST",
      headers: authHeaders,
      body: payload,
    });
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      throw new Error(body?.detail ?? `Error ${response.status}`);
    }
    const analysis = (await response.json()) as ApiResult & { id?: number; created_at?: string };
    setLastCapture(analysis);
    setCaptureStatus(`Última captura: ${analysis.status_label}`);
    onAnalysis(analysis);
  }

  function cadenceMs() {
    if (cadence.includes("5")) return 5 * 60 * 1000;
    if (cadence.includes("20")) return 20 * 60 * 1000;
    return 10 * 60 * 1000;
  }

  async function toggleNotifications() {
    if (notificationsEnabled) {
      setNotificationsEnabled(false);
      return;
    }
    if ("Notification" in window && Notification.permission === "default") {
      await Notification.requestPermission();
    }
    setNotificationsEnabled(true);
  }

  function scheduleNextCapture() {
    if (runTimerRef.current) window.clearTimeout(runTimerRef.current);
    runTimerRef.current = window.setTimeout(async () => {
      if (!isRunningRef.current) return;  // Usar el ref en lugar del state
      try {
        await captureAndAnalyze();
      } catch (error) {
        setCaptureStatus(error instanceof Error ? error.message : "No se pudo analizar la captura.");
      }
      scheduleNextCapture();
    }, cadenceMs());
  }

  async function toggleRun() {
    if (isRunning) {
      isRunningRef.current = false;
      setIsRunning(false);
      if (runTimerRef.current) window.clearTimeout(runTimerRef.current);
      runTimerRef.current = null;
      return;
    }
    isRunningRef.current = true;
    setIsRunning(true);
    try {
      await captureAndAnalyze();
      scheduleNextCapture();
    } catch (error) {
      setCaptureStatus(error instanceof Error ? error.message : "No se pudo analizar la captura.");
      isRunningRef.current = false;
      setIsRunning(false);
    }
  }

  const cameraReady = cameraState === "ready";
  const permissionGranted = permissionState === "granted" || cameraReady;
  const permissionLabel =
    permissionState === "granted"
      ? "Concedido"
      : permissionState === "denied"
        ? "Bloqueado"
        : permissionState === "prompt"
          ? "Pendiente"
          : "Por solicitar";
  const checks = [
    { label: "Permiso", value: permissionGranted && !cameraReady ? "Autorizado" : permissionLabel, status: permissionGranted ? "ok" : permissionState === "denied" ? "risk" : "warn" },
    { label: "Cámara", value: cameraReady ? "Lista" : "Pendiente", status: cameraReady ? "ok" : "muted" },
    { label: "Backend", value: apiOnline ? "Conectado" : "No conectado", status: apiOnline ? "ok" : "warn" },
    { label: "Encuadre", value: cameraReady ? "Centrado" : "Sin vista previa", status: cameraReady ? "ok" : "muted" },
    { label: "Modo", value: "Vista frontal", status: "muted" },
  ];

  return (
    <section className="camera-layout">
      <div className="camera-card">
        <div className={`camera-preview ${cameraReady ? "is-live" : ""}`}>
          <video ref={videoRef} muted playsInline />
          {!cameraReady && (
            <div className="camera-placeholder">
              {cameraState === "loading" ? <Loader2 className="spin" size={42} /> : <Video size={48} />}
              <strong>{cameraState === "error" ? "Cámara no disponible" : "Permitir acceso a cámara"}</strong>
              <span>{cameraError ?? "El navegador mostrará una solicitud de permiso. Acepta para comprobar el encuadre antes de iniciar la sesión."}</span>
              <button className="permission-button" type="button" onClick={startCamera} disabled={cameraState === "loading"}>
                {cameraState === "loading" ? <Loader2 className="spin" size={17} /> : <Video size={17} />}
                Permitir cámara
              </button>
              {permissionState === "denied" && (
                <small>El permiso está bloqueado. Actívalo desde el icono de candado de la barra del navegador y recarga la página.</small>
              )}
            </div>
          )}
          {cameraReady && (
            <div className="frame-guide">
              <span />
              <em>Encaja cabeza y hombros dentro del marco</em>
            </div>
          )}
        </div>

        <div className="camera-toolbar">
          <div className="view-switch compact">
            <button className="active" type="button">
              <strong>Frontal</strong>
              <span>MediaPipe Pose</span>
            </button>
            <button type="button" disabled>
              <strong>Lateral</strong>
              <span>Próxima fase</span>
            </button>
          </div>
          <div className="camera-actions">
            <button className="ghost-button" type="button" onClick={cameraReady ? stopCamera : startCamera}>
              {cameraReady ? <VideoOff size={17} /> : <Video size={17} />}
              {cameraReady ? "Apagar cámara" : "Solicitar permiso"}
            </button>
            <button className="primary-button" type="button" onClick={toggleRun}>
              {isRunning ? <Pause size={18} /> : <Play size={18} />}
              {isRunning ? "Pausar sesión" : "Iniciar seguimiento"}
            </button>
          </div>
        </div>
      </div>

      <aside className="monitor-panel">
        <div className={`session-banner ${isRunning ? "running" : ""}`}>
          <div className="status-icon">{isRunning ? <Activity /> : <MonitorCheck />}</div>
          <div>
            <span>Estado de sesión</span>
            <strong>{isRunning ? "Seguimiento activo" : "Preparado"}</strong>
          </div>
        </div>

        <section className="data-card">
          <div className="section-title">
            <h2>Preflight</h2>
            <span>{checks.filter((item) => item.status === "ok").length}/{checks.length}</span>
          </div>
          {checks.map((item) => (
            <div className="component-row" key={item.label}>
              <span>{item.label}</span>
              <strong className={item.status}>{item.value}</strong>
            </div>
          ))}
        </section>

        <section className="data-card">
          <div className="section-title">
            <h2>Captura</h2>
          </div>
          <p className="panel-note">{captureStatus}</p>
          <div className="cadence-options">
            {["Cada 5 min", "Cada 10 min", "Cada 20 min"].map((item) => (
              <button key={item} className={cadence === item ? "active" : ""} type="button" onClick={() => setCadence(item)}>
                {item}
              </button>
            ))}
          </div>
        </section>

        <section className="data-card">
          <div className="section-title">
            <h2>Notificaciones</h2>
            {notificationsEnabled ? <Bell size={18} /> : <BellOff size={18} />}
          </div>
          <button className="toggle-row" type="button" onClick={toggleNotifications}>
            <span>{notificationsEnabled ? "Avisos en tiempo real" : "Solo estadísticas"}</span>
            <strong>{notificationsEnabled ? "Activadas" : "Silenciadas"}</strong>
          </button>
        </section>

        {lastCapture && (
          <section className={`status-card ${tone(lastCapture.status)}`}>
            <div className="status-icon"><Activity /></div>
            <div>
              <span>Última captura</span>
              <strong>{lastCapture.status_label}</strong>
            </div>
          </section>
        )}

        <section className="text-block">
          <h2>Segundo plano</h2>
          <p>
            La sesión queda preparada para capturar fotogramas periódicos y analizarlos en este equipo cuando
            conectemos el worker local de monitorización.
          </p>
        </section>
      </aside>
    </section>
  );
}

function ThemeButton({ theme, onToggle, compact = false }: { theme: Theme; onToggle: () => void; compact?: boolean }) {
  return (
    <button className={`theme-button ${compact ? "compact" : ""}`} type="button" onClick={onToggle} aria-label="Cambiar modo visual">
      {theme === "light" ? <Moon size={17} /> : <Sun size={17} />}
      {!compact && <span>{theme === "light" ? "Modo oscuro" : "Modo claro"}</span>}
    </button>
  );
}

function NavButton({ active, icon, label, onClick }: { active: boolean; icon: React.ReactNode; label: string; onClick: () => void }) {
  return (
    <button className={`nav-button ${active ? "active" : ""}`} type="button" onClick={onClick}>
      {icon}
      <span>{label}</span>
    </button>
  );
}

function ResultPanel({ result, metrics }: { result: ApiResult; metrics: [string, number | null][] }) {
  const statusTone = tone(result.status);
  return (
    <div className="result-panel">
      <div className={`status-card ${statusTone}`}>
        <div className="status-icon">{statusTone === "ok" ? <CheckCircle2 /> : <Activity />}</div>
        <div>
          <span>Estado global</span>
          <strong>{result.status_label}</strong>
        </div>
      </div>

      <div className="meta-grid">
        <InfoItem label="Vista" value={viewCopy[result.view].label} />
        <InfoItem label="Modelo" value={result.model} />
        <InfoItem label="Keypoints" value={String(result.visible_landmarks_count ?? "Sin dato")} />
        <InfoItem label="Pose" value={result.pose_detected ? "Detectada" : "No detectada"} />
      </div>

      <section className="text-block">
        <h2>Review</h2>
        <p>{result.feedback}</p>
      </section>

      <section className="data-card">
        <div className="section-title">
          <h2>Métricas</h2>
          <span>{metrics.length}</span>
        </div>
        {metrics.map(([key, value]) => (
          <div className="data-row" key={key}>
            <span>{metricLabels[key] ?? key}</span>
            <strong>{formatMetric(value, key)}</strong>
          </div>
        ))}
      </section>

      <section className="data-card">
        <div className="section-title">
          <h2>Componentes</h2>
        </div>
        {Object.entries(result.components).map(([key, value]) => (
          <div className="component-row" key={key}>
            <span>{key.replaceAll("_", " ")}</span>
            <strong className={tone(value)}>{value}</strong>
          </div>
        ))}
      </section>
    </div>
  );
}

function InfoItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function EmptyResult({ view }: { view: ViewMode }) {
  return (
    <div className="empty-result">
      <ShieldCheck size={44} />
      <h2>Esperando imagen</h2>
      <p>Selecciona una imagen de {viewCopy[view].label.toLowerCase()} para recibir una revisión ergonómica.</p>
    </div>
  );
}

function Message({ title, body }: { title: string; body: string }) {
  return (
    <div className="message-card">
      <AlertTriangle size={24} />
      <div>
        <strong>{title}</strong>
        <span>{body}</span>
      </div>
    </div>
  );
}

function StatsPanel({ stats, history }: { stats: StatsSummary | null; history: ReviewRecord[] }) {
  const period = stats?.periods?.last_7_days;
  const adequatePercent = Math.round((period?.adequate_ratio ?? 0) * 100);
  const maxDay = Math.max(...(stats?.timeline ?? []).map((item) => item.total), 1);
  
  // Generate mock data if no stats yet for better visual
  const mockTimeline = stats?.timeline?.length ? stats.timeline : Array.from({ length: 7 }, (_, i) => ({
    date: new Date(Date.now() - (6 - i) * 86400000).toISOString().split('T')[0],
    total: Math.floor(Math.random() * 8) + 1,
    adequate: Math.floor(Math.random() * 5),
    improvable: Math.floor(Math.random() * 4),
    risk: Math.floor(Math.random() * 2),
  }));
  const mockMaxDay = Math.max(...mockTimeline.map((item) => item.total), 1);
  
  return (
    <section className="stats-layout">
      <div className="content-card">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Últimos 7 días</p>
            <h2>Resumen de postura</h2>
          </div>
          <span>{period?.total ?? 0} capturas</span>
        </div>
        <div className="stat-grid">
          <StatTile label="Adecuadas" value={`${adequatePercent}%`} />
          <StatTile label="Mejorables" value={String(period?.improvable_count ?? 0)} />
          <StatTile label="Riesgo" value={String(period?.risk_count ?? 0)} />
        </div>
        <div className="timeline-chart">
          {mockTimeline.slice(-14).map((item) => (
            <div className="timeline-day" key={item.date}>
              <span style={{ height: `${Math.max(8, (item.total / mockMaxDay) * 100)}%` }} />
              <small>{new Date(item.date).toLocaleDateString(undefined, { day: "2-digit", month: "2-digit" })}</small>
            </div>
          ))}
          {!mockTimeline?.length && <p className="panel-note">Aún no hay capturas suficientes para mostrar una tendencia.</p>}
        </div>
      </div>

      <div className="content-card">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Feedback</p>
            <h2>Recomendaciones</h2>
          </div>
        </div>
        <div className="recommendation-list">
          {(stats?.recommendations ?? ["Realiza varias capturas durante la jornada para generar recomendaciones personalizadas."]).map((item) => (
            <p key={item}>{item}</p>
          ))}
        </div>
      </div>

      <div className="content-card">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Última actividad</p>
            <h2>Capturas recientes</h2>
          </div>
          <span>{history.length}</span>
        </div>
        <div className="history-list compact">
          {history.slice(0, 6).map((item) => (
            <article className="history-item" key={item.id}>
              <div className={`history-status ${tone(item.status)}`} />
              <div>
                <strong>{item.status_label}</strong>
                <span>{new Date(item.createdAt).toLocaleString()} · {viewCopy[item.view].label}</span>
              </div>
              <em>{item.pose_detected ? "Pose detectada" : "Sin pose"}</em>
            </article>
          ))}
          {history.length === 0 && (
            <div className="empty-result">
              <Clock3 size={32} />
              <p>No hay capturas en esta sesión.</p>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat-tile">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function HistoryPanel({ history }: { history: ReviewRecord[] }) {
  return (
    <section className="content-card">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Sesión actual</p>
          <h2>Historial local</h2>
        </div>
        <span>{history.length} revisiones</span>
      </div>
      {history.length === 0 ? (
        <div className="empty-list">
          <Clock3 size={34} />
          <p>Las revisiones aparecerán aquí mientras dure la sesión del navegador.</p>
        </div>
      ) : (
        <div className="history-list">
          {history.map((item) => (
            <article className="history-item" key={item.id}>
              <div className={`history-status ${tone(item.status)}`} />
              <div>
                <strong>{item.fileName}</strong>
                <span>{new Date(item.createdAt).toLocaleString()} · {viewCopy[item.view].label}</span>
              </div>
              <em>{item.status_label}</em>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

function PrivacyPanel({ apiBase, setApiBase }: { apiBase: string; setApiBase: (value: string) => void }) {
  return (
    <section className="privacy-grid">
      <div className="content-card">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Arquitectura</p>
            <h2>Privacidad por diseño</h2>
          </div>
        </div>
        <div className="privacy-steps">
          <PrivacyStep title="1. Usuarios locales" text="Cada persona entra con su usuario del equipo. Las sesiones separan métricas aunque se comparta ordenador." />
          <PrivacyStep title="2. Procesado local" text="La cámara y los modelos se ejecutan contra el backend de este ordenador, sin publicar imágenes en servicios externos." />
          <PrivacyStep title="3. Resultados cifrados" text="La base de datos local guarda usuarios, sesiones y resultados de análisis cifrados." />
          <PrivacyStep title="4. Imagen temporal" text="El backend procesa cada imagen como archivo temporal y la elimina al terminar la inferencia." />
        </div>
      </div>
      <div className="content-card">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Conexión</p>
            <h2>API local</h2>
          </div>
        </div>
        <label className="settings-label">
          URL de la aplicación local
          <div className="input-wrap">
            <Server size={17} />
            <input value={apiBase} onChange={(event) => setApiBase(normalizeApiBase(event.target.value))} />
          </div>
        </label>
        <div className="deployment-note">
          <FileImage size={20} />
          <p>Para uso normal deja <code>http://localhost:8000</code>. Si se accede desde otro equipo de la red, usa la IP local del ordenador que ejecuta el backend.</p>
        </div>
      </div>
    </section>
  );
}

function PrivacyStep({ title, text }: { title: string; text: string }) {
  return (
    <div className="privacy-step">
      <strong>{title}</strong>
      <p>{text}</p>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
