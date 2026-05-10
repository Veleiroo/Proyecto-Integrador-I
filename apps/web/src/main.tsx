import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Bell,
  BellOff,
  CheckCircle2,
  ChevronRight,
  Clock3,
  DatabaseZap,
  EyeOff,
  FileImage,
  Info,
  KeyRound,
  Loader2,
  LockKeyhole,
  LogOut,
  Moon,
  MonitorCheck,
  Pause,
  Play,
  Server,
  ShieldCheck,
  Sun,
  TrendingUp,
  UserRound,
  Video,
  VideoOff,
  X,
} from "lucide-react";
import "./styles.css";

type ViewMode = "front" | "lateral" | "combined";
type Status = "adequate" | "improvable" | "risk" | "insufficient_data";
type Section = "camera" | "stats" | "history" | "privacy" | "debug";
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

type DevDebugResponse = {
  ok: boolean;
  result?: ApiResult;
  debug?: {
    saved_dir: string;
    original_path: string;
    annotated_path: string;
    original_preview_data_url: string;
    annotated_preview_data_url: string;
    keypoints: Array<{ name: string; x: number; y: number; visibility: number | null }>;
    rule_lines: Array<{ label: string; from: string; to: string; status: string; points: Array<[number, number]> }>;
  };
  error?: {
    type: string;
    message: string;
  };
};

const DEFAULT_API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const sectionCopy: Record<Section, { eyebrow: string; title: string }> = {
  camera: {
    eyebrow: "Monitor local",
    title: "Cámara y seguimiento ergonómico",
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
  debug: {
    eyebrow: "Perfil técnico",
    title: "Depuración visual del pipeline",
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
  combined: {
    label: "Evaluación combinada",
    short: "Frontal base con lateral opcional",
    endpoint: "/api/analyze/combined",
    model: "MediaPipe Pose + YOLO Pose",
  },
};

const metricLabels: Record<string, string> = {
  shoulder_tilt_deg: "Inclinación de hombros",
  shoulder_height_diff_ratio: "Desnivel de hombros",
  head_lateral_offset_ratio: "Desplazamiento lateral de cabeza",
  neck_tilt_deg: "Inclinación cervical frontal",
  trunk_tilt_deg: "Inclinación frontal del tronco",
  left_elbow_angle_deg: "Codo izquierdo",
  right_elbow_angle_deg: "Codo derecho",
  head_forward_offset_ratio: "Cabeza adelantada",
  neck_forward_tilt_deg: "Inclinación cervical lateral",
  trunk_forward_tilt_deg: "Inclinación lateral del tronco",
  shoulder_hip_offset_ratio: "Alineación hombro-cadera",
  lateral_elbow_angle_deg: "Codo lateral",
};

const componentLabels: Record<string, string> = {
  shoulder_tilt_status: "Inclinación de hombros",
  shoulder_height_status: "Desnivel de hombros",
  neck_status: "Cuello",
  neck_tilt_status: "Inclinación cervical",
  shoulder_status: "Hombros",
  trunk_status: "Tronco",
  trunk_tilt_status: "Inclinación del tronco",
  head_status: "Cabeza y cuello",
  head_offset_status: "Desplazamiento de cabeza",
  head_neck_status: "Cabeza y cuello",
  elbow_status: "Codos",
  left_elbow_status: "Codo izquierdo",
  right_elbow_status: "Codo derecho",
  shoulder_hip_status: "Alineación hombro-cadera",
  lateral_elbow_status: "Codo lateral",
  front_shoulder_tilt_status: "Frontal: inclinación de hombros",
  front_shoulder_height_status: "Frontal: desnivel de hombros",
  front_neck_status: "Frontal: cuello",
  front_neck_tilt_status: "Frontal: inclinación cervical",
  front_shoulder_status: "Frontal: hombros",
  front_trunk_status: "Frontal: tronco",
  front_head_status: "Frontal: cabeza y cuello",
  front_head_offset_status: "Frontal: desplazamiento de cabeza",
  front_left_elbow_status: "Frontal: codo izquierdo",
  front_right_elbow_status: "Frontal: codo derecho",
  lateral_neck_status: "Lateral: cuello",
  lateral_head_neck_status: "Lateral: cabeza y cuello",
  lateral_head_offset_status: "Lateral: cabeza adelantada",
  lateral_trunk_tilt_status: "Lateral: inclinación del tronco",
  lateral_shoulder_hip_status: "Lateral: alineación hombro-cadera",
  lateral_trunk_status: "Lateral: tronco",
  lateral_lateral_elbow_status: "Lateral: codo",
};

const statusLabels: Record<string, string> = {
  adequate: "Adecuada",
  improvable: "Mejorable",
  risk: "Riesgo",
  insufficient_data: "Datos insuficientes",
};

const backendTermLabels: Record<string, string> = {
  head: "cabeza",
  neck: "cuello",
  shoulder: "hombro",
  shoulders: "hombros",
  trunk: "tronco",
  elbow: "codo",
  hip: "cadera",
  left: "izquierdo",
  right: "derecho",
  lateral: "lateral",
  forward: "adelantado",
  tilt: "inclinación",
  offset: "desplazamiento",
  height: "altura",
  diff: "diferencia",
  angle: "ángulo",
};

function normalizeApiBase(value: string) {
  return value.trim().replace(/\/+$/, "");
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

function humanizeBackendKey(key: string) {
  return key
    .replace(/_(status|deg|ratio)$/g, "")
    .split("_")
    .map((term) => backendTermLabels[term] ?? term)
    .join(" ")
    .replace(/^./, (letter) => letter.toUpperCase());
}

function labelMetric(key: string) {
  return metricLabels[key] ?? humanizeBackendKey(key);
}

function labelComponent(key: string) {
  return componentLabels[key] ?? humanizeBackendKey(key);
}

function labelStatus(status: string) {
  return statusLabels[status] ?? humanizeBackendKey(status);
}

function formatDateTime(value: string) {
  return new Date(value).toLocaleString("es-ES", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatShortDate(value: string) {
  return new Date(value).toLocaleDateString("es-ES", {
    day: "2-digit",
    month: "2-digit",
  });
}

function isBodyComponent(key: string) {
  return key !== "overall_status";
}

function App() {
  const [session, setSession] = useState<AuthSession | null>(() => {
    const stored = localStorage.getItem("authSession");
    if (!stored) return null;
    try {
      return JSON.parse(stored) as AuthSession;
    } catch {
      localStorage.removeItem("authSession");
      return null;
    }
  });
  const [activeSection, setActiveSection] = useState<Section>("camera");
  const [theme, setTheme] = useState<Theme>((localStorage.getItem("theme") as Theme | null) ?? "light");
  const [apiBase, setApiBase] = useState(() => {
    return normalizeApiBase(localStorage.getItem("apiBase") ?? DEFAULT_API_BASE);
  });
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [selectedRecord, setSelectedRecord] = useState<ReviewRecord | null>(null);
  const [history, setHistory] = useState<ReviewRecord[]>([]);
  const [stats, setStats] = useState<StatsSummary | null>(null);
  const [notificationsEnabled, setNotificationsEnabled] = useState(() => localStorage.getItem("notificationsEnabled") !== "false");
  const refreshRequestRef = useRef(0);

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
    localStorage.setItem("notificationsEnabled", String(notificationsEnabled));
  }, [notificationsEnabled]);

  const activeCopy = sectionCopy[activeSection];
  const isDev = session?.user.role === "dev";
  const accessToken = session?.access_token;
  const authHeaders = useMemo<Record<string, string>>(
    () => {
      const headers: Record<string, string> = {};
      if (accessToken) headers.Authorization = `Bearer ${accessToken}`;
      return headers;
    },
    [accessToken],
  );

  useEffect(() => {
    if (!session) return;
    refreshUserData();
  }, [session?.access_token, apiBase]);

  async function refreshUserData() {
    if (!session) return;
    const requestId = ++refreshRequestRef.current;
    try {
      const [historyResponse, statsResponse] = await Promise.all([
        fetch(`${apiBase}/api/analyses?limit=30`, { headers: authHeaders }),
        fetch(`${apiBase}/api/summary`, { headers: authHeaders }),
      ]);
      if (requestId !== refreshRequestRef.current) return;
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
    } catch {
      setApiOnline(false);
    }
  }

  function pushAnalysis(nextResult: ApiResult & { id?: number; created_at?: string }, fileName: string) {
    const nextRecord: ReviewRecord = {
      ...nextResult,
      id: nextResult.id ?? crypto.randomUUID(),
      fileName,
      createdAt: nextResult.created_at ?? new Date().toISOString(),
    };
    setSelectedRecord(nextRecord);
    setHistory((items) => [
      nextRecord,
      ...items.slice(0, 29),
    ]);
    void refreshUserData();
  }

  function notifyAnalysis(nextResult: ApiResult) {
    if (!notificationsEnabled) return;
    const message = `${nextResult.status_label}: ${nextResult.feedback}`;
    if ("Notification" in window && Notification.permission === "granted") {
      new Notification("PostureOS", { body: message });
    }
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
    setSelectedRecord(null);
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
          <NavButton active={activeSection === "stats"} icon={<BarChart3 />} label="Estadísticas" onClick={() => setActiveSection("stats")} />
          <NavButton active={activeSection === "history"} icon={<BarChart3 />} label="Historial local" onClick={() => setActiveSection("history")} />
          <NavButton active={activeSection === "privacy"} icon={<ShieldCheck />} label="Información" onClick={() => setActiveSection("privacy")} />
          {isDev && <NavButton active={activeSection === "debug"} icon={<FileImage />} label="Debug visual" onClick={() => setActiveSection("debug")} />}
        </nav>

        {isDev && (
          <div className={`connection-card ${apiOnline ? "online" : "offline"}`}>
            <Server size={18} />
            <div>
              <strong>{apiOnline ? "Backend conectado" : "Backend no disponible"}</strong>
              <span>{apiBase}</span>
            </div>
          </div>
        )}

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
        {activeSection === "history" && <HistoryPanel history={history} onSelect={setSelectedRecord} />}
        {activeSection === "stats" && <StatsPanel stats={stats} history={history} onSelect={setSelectedRecord} />}
        {activeSection === "privacy" && <PrivacyPanel apiBase={apiBase} setApiBase={setApiBase} isDev={isDev} />}
        {activeSection === "debug" && isDev && <DevDebugPanel apiBase={apiBase} authHeaders={authHeaders} />}
      </section>
      {selectedRecord && <AnalysisDetail record={selectedRecord} onClose={() => setSelectedRecord(null)} />}
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
              <span>Ergonomía local privada</span>
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
          <details className="advanced-login">
            <summary>Conexión avanzada</summary>
            <label>
              API local
              <div className="input-wrap">
                <Server size={17} />
                <input value={apiBase} onChange={(event) => setApiBase(normalizeApiBase(event.target.value))} />
              </div>
            </label>
          </details>
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
  const frontVideoRef = useRef<HTMLVideoElement | null>(null);
  const lateralVideoRef = useRef<HTMLVideoElement | null>(null);
  const frontStreamRef = useRef<MediaStream | null>(null);
  const lateralStreamRef = useRef<MediaStream | null>(null);
  const runTimerRef = useRef<number | null>(null);
  const isRunningRef = useRef(false);
  const trackingViewRef = useRef<Extract<ViewMode, "front" | "lateral">>("front");
  const [cameraStates, setCameraStates] = useState<Record<Extract<ViewMode, "front" | "lateral">, "idle" | "loading" | "ready" | "error">>({
    front: "idle",
    lateral: "idle",
  });
  const [permissionState, setPermissionState] = useState<PermissionState | "unsupported">("unsupported");
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [frontDeviceId, setFrontDeviceId] = useState("");
  const [lateralDeviceId, setLateralDeviceId] = useState("");
  const [activeView, setActiveView] = useState<Extract<ViewMode, "front" | "lateral">>("front");
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [cadence, setCadence] = useState("Cada 1 min");
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
      stopAllCameras();
    };
  }, []);

  useEffect(() => {
    refreshDevices();
    navigator.mediaDevices?.addEventListener?.("devicechange", refreshDevices);
    return () => navigator.mediaDevices?.removeEventListener?.("devicechange", refreshDevices);
  }, []);

  useEffect(() => {
    if (frontStreamRef.current) stopCamera("front");
  }, [frontDeviceId]);

  useEffect(() => {
    if (lateralStreamRef.current) stopCamera("lateral");
  }, [lateralDeviceId]);

  async function refreshDevices() {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    const nextDevices = (await navigator.mediaDevices.enumerateDevices()).filter((device) => device.kind === "videoinput");
    setDevices(nextDevices);
    setFrontDeviceId((current) => current || nextDevices[0]?.deviceId || "");
    setLateralDeviceId((current) => current || nextDevices[1]?.deviceId || "");
  }

  function videoFor(view: Extract<ViewMode, "front" | "lateral">) {
    return view === "front" ? frontVideoRef.current : lateralVideoRef.current;
  }

  function streamFor(view: Extract<ViewMode, "front" | "lateral">) {
    return view === "front" ? frontStreamRef.current : lateralStreamRef.current;
  }

  function setStreamFor(view: Extract<ViewMode, "front" | "lateral">, stream: MediaStream | null) {
    if (view === "front") {
      frontStreamRef.current = stream;
    } else {
      lateralStreamRef.current = stream;
    }
  }

  async function startCamera(view: Extract<ViewMode, "front" | "lateral"> = activeView) {
    const video = videoFor(view);
    if (streamFor(view) && video?.srcObject) {
      setCameraStates((current) => ({ ...current, [view]: "ready" }));
      return true;
    }
    const deviceId = view === "front" ? frontDeviceId : lateralDeviceId;
    setCameraStates((current) => ({ ...current, [view]: "loading" }));
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: "user" }),
        },
        audio: false,
      });
      setStreamFor(view, stream);
      if (video) {
        video.srcObject = stream;
        await video.play();
      }
      setCameraStates((current) => ({ ...current, [view]: "ready" }));
      setPermissionState("granted");
      await refreshDevices();
      return true;
    } catch (error) {
      setCameraStates((current) => ({ ...current, [view]: "error" }));
      if (error instanceof DOMException && error.name === "NotAllowedError") {
        setPermissionState("denied");
      }
      setCameraError(error instanceof Error ? error.message : "No se pudo abrir la cámara.");
      return false;
    }
  }

  function stopCamera(view: Extract<ViewMode, "front" | "lateral"> = activeView) {
    streamFor(view)?.getTracks().forEach((track) => track.stop());
    setStreamFor(view, null);
    const video = videoFor(view);
    if (video) video.srcObject = null;
    setCameraStates((current) => ({ ...current, [view]: "idle" }));
  }

  function stopAllCameras() {
    stopCamera("front");
    stopCamera("lateral");
  }

  function snapshotVideo(video: HTMLVideoElement | null) {
    if (!video) return null;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    const context = canvas.getContext("2d");
    if (!context) return null;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
  }

  async function captureFrameBlob(view: Extract<ViewMode, "front" | "lateral">, stopAfter = true) {
    const started = await startCamera(view);
    if (!started) return null;
    await new Promise((resolve) => window.setTimeout(resolve, 900));
    const blob = await snapshotVideo(videoFor(view));
    if (stopAfter) stopCamera(view);
    return blob;
  }

  async function postAnalysis(view: Extract<ViewMode, "front" | "lateral">, blob: Blob) {
    const payload = new FormData();
    payload.append("file", blob, `${view}-capture-${Date.now()}.jpg`);
    const response = await fetch(`${apiBase}${viewCopy[view].endpoint}`, {
      method: "POST",
      headers: authHeaders,
      body: payload,
    });
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      setCaptureStatus(body?.detail ?? `Error ${response.status}`);
      throw new Error(body?.detail ?? `Error ${response.status}`);
    }
    return (await response.json()) as ApiResult & { id?: number; created_at?: string };
  }

  async function postCombinedAnalysis(frontBlob: Blob, lateralBlob: Blob) {
    const payload = new FormData();
    payload.append("front_file", frontBlob, `front-combined-${Date.now()}.jpg`);
    payload.append("lateral_file", lateralBlob, `lateral-combined-${Date.now()}.jpg`);
    const response = await fetch(`${apiBase}/api/analyze/combined`, {
      method: "POST",
      headers: authHeaders,
      body: payload,
    });
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      setCaptureStatus(body?.detail ?? `Error ${response.status}`);
      throw new Error(body?.detail ?? `Error ${response.status}`);
    }
    return (await response.json()) as ApiResult & { id?: number; created_at?: string };
  }

  async function captureAndAnalyze(view: Extract<ViewMode, "front" | "lateral"> = "front") {
    if (isCapturing) return false;
    setIsCapturing(true);
    setCameraError(null);
    try {
      if (view === "lateral") {
        return await captureDualAndAnalyze();
      }
      setCaptureStatus(`Capturando ${viewCopy[view].label.toLowerCase()}...`);
      const blob = await captureFrameBlob(view);
      if (!blob) {
        setCaptureStatus("No se pudo capturar una imagen válida.");
        return false;
      }
      setCaptureStatus(`Analizando ${viewCopy[view].label.toLowerCase()}...`);
      const analysis = await postAnalysis(view, blob);
      setLastCapture(analysis);
      setCaptureStatus(`Última captura: ${analysis.status_label}`);
      onAnalysis(analysis);
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : "No se pudo analizar la captura.";
      setCaptureStatus(message);
      setCameraError(message);
      return false;
    } finally {
      setIsCapturing(false);
    }
  }

  async function captureDualAndAnalyze() {
    setCaptureStatus("Capturando frontal y lateral...");
    const [frontStarted, lateralStarted] = await Promise.all([startCamera("front"), startCamera("lateral")]);
    if (!frontStarted || !lateralStarted) {
      setCaptureStatus("No se pudieron activar las dos cámaras para la evaluación combinada.");
      return false;
    }
    await new Promise((resolve) => window.setTimeout(resolve, 900));
    const [frontBlob, lateralBlob] = await Promise.all([snapshotVideo(frontVideoRef.current), snapshotVideo(lateralVideoRef.current)]);
    stopAllCameras();
    if (!frontBlob || !lateralBlob) {
      setCaptureStatus("No se pudieron capturar las dos vistas.");
      return false;
    }

    setCaptureStatus("Analizando frontal + lateral...");
    const combined = await postCombinedAnalysis(frontBlob, lateralBlob);
    setLastCapture(combined);
    setCaptureStatus(`Combinada actualizada: ${combined.status_label}`);
    onAnalysis(combined);
    return true;
  }

  function cadenceMs() {
    if (cadence.includes("1")) return 1 * 60 * 1000;
    if (cadence.includes("5")) return 5 * 60 * 1000;
    if (cadence.includes("20")) return 20 * 60 * 1000;
    return 1 * 60 * 1000;
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

  function scheduleNextCapture(view: Extract<ViewMode, "front" | "lateral">) {
    if (runTimerRef.current) window.clearTimeout(runTimerRef.current);
    runTimerRef.current = window.setTimeout(async () => {
      if (!isRunningRef.current) return;
      const ok = await captureAndAnalyze(view);
      if (!ok) {
        isRunningRef.current = false;
        setIsRunning(false);
        return;
      }
      scheduleNextCapture(view);
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
    const nextTrackingView = activeView;
    trackingViewRef.current = nextTrackingView;
    isRunningRef.current = true;
    setIsRunning(true);
    try {
      const ok = await captureAndAnalyze(nextTrackingView);
      if (ok) {
        scheduleNextCapture(nextTrackingView);
      } else {
        isRunningRef.current = false;
        setIsRunning(false);
      }
    } catch (error) {
      setCaptureStatus(error instanceof Error ? error.message : "No se pudo analizar la captura.");
      isRunningRef.current = false;
      setIsRunning(false);
    }
  }

  const frontReady = cameraStates.front === "ready";
  const lateralReady = cameraStates.lateral === "ready";
  const cameraBusy = isCapturing || cameraStates.front === "loading" || cameraStates.lateral === "loading";
  const trackingLabel = activeView === "front" ? "Seguimiento frontal" : "Seguimiento doble";
  const runningLabel = trackingViewRef.current === "front" ? "Seguimiento frontal activo" : "Seguimiento doble activo";
  const cameraReady = activeView === "front" ? frontReady : frontReady && lateralReady;
  const permissionGranted = permissionState === "granted" || frontReady || lateralReady;
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
    { label: "Encuadre", value: cameraReady ? (activeView === "front" ? "Frontal" : "Doble vista") : "Sin vista previa", status: cameraReady ? "ok" : "muted" },
    { label: "Modo", value: activeView === "front" ? "Frontal base" : "Frontal + lateral", status: "muted" },
  ];

  return (
    <section className="camera-layout">
      <div className="camera-card">
        <div className={activeView === "lateral" ? "dual-camera-grid" : ""}>
          <div className={`camera-preview ${frontReady ? "is-live" : ""} ${activeView === "lateral" ? "compact" : ""}`}>
            {activeView === "lateral" && <div className="camera-slot-label">Frontal</div>}
            <video ref={frontVideoRef} muted playsInline />
            {!frontReady && (
              <div className="camera-placeholder">
                {cameraStates.front === "loading" ? <Loader2 className="spin" size={42} /> : <Video size={48} />}
                <strong>{cameraStates.front === "error" ? "Cámara frontal no disponible" : "Permitir cámara frontal"}</strong>
                <span>{cameraError ?? "La vista frontal sigue siendo la referencia base para el seguimiento."}</span>
                <button className="permission-button" type="button" onClick={() => startCamera("front")} disabled={cameraStates.front === "loading"}>
                  {cameraStates.front === "loading" ? <Loader2 className="spin" size={17} /> : <Video size={17} />}
                  Permitir frontal
                </button>
                {permissionState === "denied" && (
                  <small>El permiso está bloqueado. Actívalo desde el icono de candado de la barra del navegador y recarga la página.</small>
                )}
              </div>
            )}
            {frontReady && (
              <div className="frame-guide">
                <span />
                <em>Encaja cabeza y hombros dentro del marco</em>
              </div>
            )}
          </div>

          {activeView === "lateral" && (
            <div className={`camera-preview compact ${lateralReady ? "is-live" : ""}`}>
              <div className="camera-slot-label">Lateral</div>
              <video ref={lateralVideoRef} muted playsInline />
              {!lateralReady && (
                <div className="camera-placeholder">
                  {cameraStates.lateral === "loading" ? <Loader2 className="spin" size={42} /> : <Video size={48} />}
                  <strong>{cameraStates.lateral === "error" ? "Cámara lateral no disponible" : "Permitir cámara lateral"}</strong>
                  <span>{cameraError ?? "Coloca el móvil o segunda cámara de perfil para medir tronco, cabeza adelantada y codo."}</span>
                  <button className="permission-button" type="button" onClick={() => startCamera("lateral")} disabled={cameraStates.lateral === "loading"}>
                    {cameraStates.lateral === "loading" ? <Loader2 className="spin" size={17} /> : <Video size={17} />}
                    Permitir lateral
                  </button>
                </div>
              )}
              {lateralReady && (
                <div className="frame-guide">
                  <span />
                  <em>Coloca el cuerpo de perfil dentro del marco</em>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="camera-toolbar">
          <div className="view-switch compact">
            <button className={activeView === "front" ? "active" : ""} type="button" onClick={() => setActiveView("front")}>
              <strong>Frontal</strong>
              <span>Base de seguimiento</span>
            </button>
            <button className={activeView === "lateral" ? "active" : ""} type="button" onClick={() => setActiveView("lateral")}>
              <strong>Lateral</strong>
              <span>Opcional + combinado</span>
            </button>
          </div>
          <div className="camera-actions">
            <button className="ghost-button" type="button" onClick={cameraReady ? stopAllCameras : () => activeView === "lateral" ? Promise.all([startCamera("front"), startCamera("lateral")]) : startCamera("front")}>
              {cameraReady ? <VideoOff size={17} /> : <Video size={17} />}
              {cameraReady ? "Apagar cámaras" : "Solicitar permiso"}
            </button>
            <button className="ghost-button" type="button" onClick={() => captureAndAnalyze(activeView)} disabled={cameraBusy}>
              {isCapturing ? <Loader2 className="spin" size={17} /> : <Activity size={17} />}
              {activeView === "front" ? "Capturar frontal" : "Capturar doble vista"}
            </button>
            <button className="primary-button" type="button" onClick={toggleRun} disabled={isCapturing && !isRunning}>
              {isRunning ? <Pause size={18} /> : <Play size={18} />}
              {isRunning ? "Pausar sesión" : trackingLabel}
            </button>
          </div>
        </div>
      </div>

      <aside className="monitor-panel">
        <div className={`session-banner ${isRunning ? "running" : ""}`}>
          <div className="status-icon">{isRunning ? <Activity /> : <MonitorCheck />}</div>
          <div>
            <span>Estado de sesión</span>
            <strong>{isRunning ? runningLabel : "Preparado"}</strong>
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
            <h2>Cámara</h2>
            <Video size={18} />
          </div>
          <label className="settings-label compact">
            Cámara frontal
            <select value={frontDeviceId} onChange={(event) => setFrontDeviceId(event.target.value)}>
              {devices.length === 0 ? (
                <option value="">Cámara predeterminada</option>
              ) : (
                devices.map((device, index) => (
                  <option key={device.deviceId || index} value={device.deviceId}>
                    {device.label || `Cámara ${index + 1}`}
                  </option>
                ))
              )}
            </select>
          </label>
          {activeView === "lateral" && (
            <label className="settings-label compact">
              Cámara lateral
              <select value={lateralDeviceId} onChange={(event) => setLateralDeviceId(event.target.value)}>
                <option value="">Cámara predeterminada</option>
                {devices.map((device, index) => (
                  <option key={device.deviceId || index} value={device.deviceId}>
                    {device.label || `Cámara ${index + 1}`}
                  </option>
                ))}
              </select>
            </label>
          )}
        </section>

        <section className="data-card">
          <div className="section-title">
            <h2>Captura</h2>
          </div>
          <p className="panel-note">{captureStatus}</p>
          <div className="cadence-options">
            {["Cada 1 min", "Cada 5 min", "Cada 20 min"].map((item) => (
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
          <section className="data-card last-result-card">
            <div className="section-title">
              <h2>Última evaluación</h2>
              <span>{viewCopy[lastCapture.view].label}</span>
            </div>
            <div className={`status-card ${tone(lastCapture.status)}`}>
              <div className="status-icon"><Activity /></div>
              <div>
                <span>{lastCapture.model}</span>
                <strong>{lastCapture.status_label}</strong>
              </div>
            </div>
            <p className="panel-note">{lastCapture.feedback}</p>
            <div className="mini-metric-list">
              {Object.entries(lastCapture.metrics ?? {}).filter(([, value]) => value !== null).slice(0, 4).map(([key, value]) => (
                <div className="data-row" key={key}>
                  <span>{labelMetric(key)}</span>
                  <strong>{formatMetric(value, key)}</strong>
                </div>
              ))}
            </div>
          </section>
        )}

        <section className="text-block">
          <h2>Segundo plano</h2>
          <p>
            La sesión captura fotogramas periódicos y los analiza contra el backend local sin guardar las imágenes.
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
        <h2>Evaluación</h2>
        <p>{result.feedback}</p>
      </section>

      <section className="data-card">
        <div className="section-title">
          <h2>Métricas</h2>
          <span>{metrics.length}</span>
        </div>
        {metrics.map(([key, value]) => (
          <div className="data-row" key={key}>
            <span>{labelMetric(key)}</span>
            <strong>{formatMetric(value, key)}</strong>
          </div>
        ))}
      </section>

      <section className="data-card">
        <div className="section-title">
          <h2>Componentes</h2>
        </div>
        {Object.entries(result.components).filter(([key]) => isBodyComponent(key)).map(([key, value]) => (
          <div className="component-row" key={key}>
            <span>{labelComponent(key)}</span>
            <strong className={tone(value)}>{labelStatus(value)}</strong>
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

function metricTrendLabel(delta: number | null, key: string) {
  if (delta === null) return "Sin tendencia";
  const formatted = formatMetric(Math.abs(delta), key);
  if (Math.abs(delta) < 0.01) return "Estable";
  return delta > 0 ? `Sube ${formatted}` : `Baja ${formatted}`;
}

function buildHistoricRecommendations(history: ReviewRecord[], selectedMetric: string | null) {
  if (history.length === 0) {
    return ["Realiza varias capturas durante la jornada para detectar patrones reales antes de tomar decisiones."];
  }
  const recent = history.slice(0, 20);
  const riskCount = recent.filter((item) => item.status === "risk").length;
  const improvableCount = recent.filter((item) => item.status === "improvable").length;
  const componentCounts = new Map<string, number>();
  recent.forEach((item) => {
    Object.entries(item.components).forEach(([component, status]) => {
      if (isBodyComponent(component) && (status === "risk" || status === "improvable")) {
        componentCounts.set(component, (componentCounts.get(component) ?? 0) + 1);
      }
    });
  });
  const topComponent = [...componentCounts.entries()].sort((a, b) => b[1] - a[1])[0];
  const recommendations: string[] = [];
  if (topComponent) {
    recommendations.push(
      `${labelComponent(topComponent[0])} aparece como foco recurrente en ${topComponent[1]} de las últimas ${recent.length} capturas.`,
    );
  }
  if (riskCount > 0) {
    recommendations.push(`Hay ${riskCount} ${riskCount === 1 ? "captura reciente" : "capturas recientes"} en riesgo: conviene revisar el puesto antes de alargar la sesión.`);
  } else if (improvableCount > 0) {
    recommendations.push(`Predominan avisos mejorables (${improvableCount}). Ajustes pequeños y constantes deberían estabilizar la tendencia.`);
  } else {
    recommendations.push("Las capturas recientes son estables. Mantén pausas cortas y evita sostener la misma postura demasiado tiempo.");
  }
  if (selectedMetric) {
    const points = history
      .filter((item) => typeof item.metrics[selectedMetric] === "number")
      .slice()
      .reverse();
    if (points.length >= 2) {
      const first = points[0].metrics[selectedMetric] as number;
      const last = points[points.length - 1].metrics[selectedMetric] as number;
      const label = labelMetric(selectedMetric);
      const delta = last - first;
      if (Math.abs(delta) >= 0.01) {
        recommendations.push(`${label}: ${delta > 0 ? "empeora o aumenta" : "mejora o disminuye"} respecto a la primera medición disponible (${formatMetric(Math.abs(delta), selectedMetric)}).`);
      }
    }
  }
  return recommendations.slice(0, 4);
}

function StatsPanel({ stats, history, onSelect }: { stats: StatsSummary | null; history: ReviewRecord[]; onSelect: (record: ReviewRecord) => void }) {
  const period = stats?.periods?.last_7_days;
  const periodTotal = period?.total ?? 0;
  const adequatePercent = Math.round((period?.adequate_ratio ?? 0) * 100);
  const improvablePercent = periodTotal > 0 ? Math.round(((period?.improvable_count ?? 0) / periodTotal) * 100) : 0;
  const riskPercent = periodTotal > 0 ? Math.round(((period?.risk_count ?? 0) / periodTotal) * 100) : 0;
  const timeline = stats?.timeline ?? [];
  const maxDay = Math.max(...timeline.map((item) => item.total), 1);
  const metricOptions = useMemo(
    () => [...new Set(history.flatMap((item) => Object.entries(item.metrics).filter(([, value]) => typeof value === "number").map(([key]) => key)))],
    [history],
  );
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null);
  const activeMetric = selectedMetric && metricOptions.includes(selectedMetric) ? selectedMetric : metricOptions[0] ?? null;
  const metricPoints = useMemo(
    () =>
      activeMetric
        ? history
            .filter((item) => typeof item.metrics[activeMetric] === "number")
            .slice()
            .reverse()
            .slice(-14)
        : [],
    [activeMetric, history],
  );
  const metricValues = metricPoints.map((item) => item.metrics[activeMetric ?? ""] as number);
  const minMetric = Math.min(...metricValues, 0);
  const maxMetric = Math.max(...metricValues, 1);
  const metricRange = maxMetric - minMetric || 1;
  const metricDelta = metricValues.length >= 2 ? metricValues[metricValues.length - 1] - metricValues[0] : null;
  const recommendations = buildHistoricRecommendations(history, activeMetric);

  return (
    <section className="stats-layout">
      <div className="content-card">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Últimos 7 días</p>
            <h2>Resumen de postura</h2>
          </div>
          <span>{periodTotal} capturas</span>
        </div>
        <div className="stat-grid">
          <StatTile label="Adecuadas" value={`${adequatePercent}%`} />
          <StatTile label="Mejorables" value={`${improvablePercent}%`} />
          <StatTile label="Riesgo" value={`${riskPercent}%`} />
        </div>
        <div className="timeline-chart">
          <TimelineChart timeline={timeline.slice(-14)} maxDay={maxDay} />
          {timeline.length === 0 && <p className="panel-note">Aún no hay capturas suficientes para mostrar una tendencia.</p>}
        </div>
        {timeline.length > 0 && <StatusLegend />}
      </div>

      <div className="content-card">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Tendencia corporal</p>
            <h2>Evolución por métrica</h2>
          </div>
          <span>{activeMetric ? metricTrendLabel(metricDelta, activeMetric) : "Sin datos"}</span>
        </div>
        <div className="metric-selector">
          {metricOptions.length > 0 ? (
            metricOptions.map((key) => (
              <button className={activeMetric === key ? "active" : ""} key={key} type="button" onClick={() => setSelectedMetric(key)}>
                {labelMetric(key)}
              </button>
            ))
          ) : (
            <p className="panel-note">Cuando haya métricas suficientes podrás elegir la parte del cuerpo a analizar.</p>
          )}
        </div>
        <div className="trend-chart">
          <MetricTrendChart activeMetric={activeMetric} maxMetric={maxMetric} metricPoints={metricPoints} metricRange={metricRange} minMetric={minMetric} onSelect={onSelect} />
          {metricPoints.length === 0 && <p className="panel-note">Aún no hay puntos suficientes para mostrar una tendencia corporal.</p>}
        </div>
        {metricPoints.length > 0 && <StatusLegend />}
      </div>

      <div className="content-card">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Feedback</p>
            <h2>Recomendaciones basadas en histórico</h2>
          </div>
        </div>
        <div className="recommendation-list">
          {recommendations.map((item) => (
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
            <button className="history-item" key={item.id} type="button" onClick={() => onSelect(item)}>
              <div className={`history-status ${tone(item.status)}`} />
              <div>
                <strong>{item.status_label}</strong>
                <span>{formatDateTime(item.createdAt)} · {viewCopy[item.view].label}</span>
              </div>
              <em>{item.pose_detected ? "Pose detectada" : "Sin pose"}</em>
            </button>
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

function TimelineChart({ timeline, maxDay }: { timeline: StatsSummary["timeline"]; maxDay: number }) {
  if (timeline.length === 0) return null;
  return (
    <div className="timeline-bars">
      {timeline.map((item) => {
        const adequateHeight = (item.adequate / maxDay) * 100;
        const improvableHeight = (item.improvable / maxDay) * 100;
        const riskHeight = (item.risk / maxDay) * 100;
        return (
          <div className="timeline-day" key={item.date}>
            <div className="stacked-bar" title={`${item.total} capturas`}>
              <span className="risk-segment" style={{ height: `${riskHeight}%` }} />
              <span className="warn-segment" style={{ height: `${improvableHeight}%` }} />
              <span className="ok-segment" style={{ height: `${adequateHeight}%` }} />
            </div>
            <small>{formatShortDate(item.date)}</small>
          </div>
        );
      })}
    </div>
  );
}

function StatusLegend() {
  return (
    <div className="status-legend" aria-label="Leyenda de estados">
      <span><i className="ok-segment" /> Adecuada</span>
      <span><i className="warn-segment" /> Mejorable</span>
      <span><i className="risk-segment" /> Riesgo</span>
    </div>
  );
}

function MetricTrendChart({
  activeMetric,
  maxMetric,
  metricPoints,
  metricRange,
  minMetric,
  onSelect,
}: {
  activeMetric: string | null;
  maxMetric: number;
  metricPoints: ReviewRecord[];
  metricRange: number;
  minMetric: number;
  onSelect: (record: ReviewRecord) => void;
}) {
  if (!activeMetric || metricPoints.length === 0) return null;
  const width = 420;
  const height = 170;
  const paddingX = 18;
  const paddingY = 18;
  const plotWidth = width - paddingX * 2;
  const plotHeight = height - paddingY * 2;
  const points = metricPoints.map((item, index) => {
    const value = item.metrics[activeMetric] as number;
    const x = paddingX + (metricPoints.length === 1 ? plotWidth / 2 : (index / (metricPoints.length - 1)) * plotWidth);
    const y = paddingY + plotHeight - ((value - minMetric) / metricRange) * plotHeight;
    return { item, value, x, y };
  });
  const path = points.map((point) => `${point.x},${point.y}`).join(" ");
  return (
    <div className="metric-trend">
      <div className="trend-scale">
        <span>{formatMetric(maxMetric, activeMetric)}</span>
        <span>{formatMetric(minMetric, activeMetric)}</span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Tendencia de ${labelMetric(activeMetric)}`}>
        <line className="trend-grid-line" x1={paddingX} x2={width - paddingX} y1={paddingY} y2={paddingY} />
        <line className="trend-grid-line" x1={paddingX} x2={width - paddingX} y1={height / 2} y2={height / 2} />
        <line className="trend-grid-line" x1={paddingX} x2={width - paddingX} y1={height - paddingY} y2={height - paddingY} />
        <polyline className="trend-line" fill="none" points={path} />
        {points.map((point) => (
          <g className="trend-dot-button" key={point.item.id} role="button" tabIndex={0} onClick={() => onSelect(point.item)} onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") onSelect(point.item);
          }}>
            <circle className={`trend-dot ${tone(point.item.status)}`} cx={point.x} cy={point.y} r="6" />
            <title>{formatMetric(point.value, activeMetric)}</title>
          </g>
        ))}
      </svg>
      <div className="trend-labels">
        {points.map((point) => (
          <button key={point.item.id} type="button" onClick={() => onSelect(point.item)}>
            {formatShortDate(point.item.createdAt)}
          </button>
        ))}
      </div>
    </div>
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

function HistoryPanel({ history, onSelect }: { history: ReviewRecord[]; onSelect: (record: ReviewRecord) => void }) {
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
            <button className="history-item" key={item.id} type="button" onClick={() => onSelect(item)}>
              <div className={`history-status ${tone(item.status)}`} />
              <div>
                <strong>{item.fileName}</strong>
                <span>{formatDateTime(item.createdAt)} · {viewCopy[item.view].label}</span>
              </div>
              <em>{item.status_label}</em>
            </button>
          ))}
        </div>
      )}
    </section>
  );
}

function DevDebugPanel({ apiBase, authHeaders }: { apiBase: string; authHeaders: Record<string, string> }) {
  const [debugView, setDebugView] = useState<Extract<ViewMode, "front" | "lateral">>("front");
  const [file, setFile] = useState<File | null>(null);
  const [response, setResponse] = useState<DevDebugResponse | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const savedFolder = response?.debug?.saved_dir.split(/[\\/]/).slice(-2).join("/");

  async function submitDebug(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file || isSubmitting) return;
    setIsSubmitting(true);
    setResponse(null);
    const payload = new FormData();
    payload.append("view", debugView);
    payload.append("file", file, file.name);
    try {
      const nextResponse = await fetch(`${apiBase}/api/dev/analyze-image`, {
        method: "POST",
        headers: authHeaders,
        body: payload,
      });
      const body = await nextResponse.json().catch(() => null);
      if (!nextResponse.ok) {
        setResponse({
          ok: false,
          error: {
            type: `HTTP ${nextResponse.status}`,
            message: body?.detail ?? "No se pudo ejecutar la depuración.",
          },
        });
        return;
      }
      setResponse(body as DevDebugResponse);
    } catch (error) {
      setResponse({
        ok: false,
        error: {
          type: error instanceof Error ? error.name : "Error",
          message: error instanceof Error ? error.message : "No se pudo conectar con el backend.",
        },
      });
    } finally {
      setIsSubmitting(false);
    }
  }

  const metrics = response?.result ? Object.entries(response.result.metrics ?? {}).filter(([, value]) => value !== null) : [];

  return (
    <section className="debug-layout">
      <form className="content-card debug-uploader" onSubmit={submitDebug}>
        <div className="section-heading">
          <div>
            <p className="eyebrow">Entrada manual</p>
            <h2>Probar imagen sin cámara</h2>
          </div>
          <span>{viewCopy[debugView].model}</span>
        </div>
        <div className="view-switch compact">
          <button className={debugView === "front" ? "active" : ""} type="button" onClick={() => setDebugView("front")}>
            <strong>Frontal</strong>
            <span>MediaPipe</span>
          </button>
          <button className={debugView === "lateral" ? "active" : ""} type="button" onClick={() => setDebugView("lateral")}>
            <strong>Lateral</strong>
            <span>YOLO Pose</span>
          </button>
        </div>
        <label className="debug-dropzone">
          <input accept="image/*" type="file" onChange={(event) => {
            setFile(event.target.files?.[0] ?? null);
            setResponse(null);
          }} />
          <FileImage size={34} />
          <strong>{file ? file.name : "Selecciona una imagen"}</strong>
          <span>El perfil dev guarda original y overlay anotado para demo y revisión técnica.</span>
        </label>
        <button className="primary-button full" type="submit" disabled={!file || isSubmitting}>
          {isSubmitting ? <Loader2 className="spin" size={18} /> : <Activity size={18} />}
          Ejecutar depuración
        </button>
        {response?.debug && (
          <div className="deployment-note">
            <DatabaseZap size={19} />
            <p title={response.debug.saved_dir}>Guardado en <code>{savedFolder}</code></p>
          </div>
        )}
      </form>

      <div className="content-card debug-output">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Salida visual</p>
            <h2>Keypoints y reglas</h2>
          </div>
          {response?.debug && <span>{response.debug.keypoints.length} puntos</span>}
        </div>
        {!response && (
          <div className="empty-result">
            <FileImage size={38} />
            <p>Sube una imagen para generar el overlay con puntos corporales y líneas de reglas.</p>
          </div>
        )}
        {response?.error && (
          <Message title={response.error.type} body={response.error.message} />
        )}
        {response?.debug && (
          <>
            <div className="debug-images">
              <figure>
                <img src={response.debug.original_preview_data_url} alt="Captura original" />
                <figcaption>Original guardada</figcaption>
              </figure>
              <figure>
                <img src={response.debug.annotated_preview_data_url} alt="Captura anotada con keypoints" />
                <figcaption>Overlay de keypoints y reglas</figcaption>
              </figure>
            </div>
            <div className="debug-grid">
              <section className="data-card">
                <div className="section-title">
                  <h2>Líneas de regla</h2>
                  <span>{response.debug.rule_lines.length}</span>
                </div>
                {response.debug.rule_lines.length === 0 ? (
                  <p className="panel-note compact">No se generaron líneas con los puntos visibles de esta captura.</p>
                ) : (
                  response.debug.rule_lines.map((line, index) => (
                    <div className="component-row" key={`${line.label}-${index}`}>
                      <span>{line.label}</span>
                      <strong className={tone(line.status)}>{labelStatus(line.status)}</strong>
                    </div>
                  ))
                )}
              </section>
              {response.result && (
                <section className="data-card">
                  <div className="section-title">
                    <h2>Resultado</h2>
                    <span>{response.result.status_label}</span>
                  </div>
                  <p className="panel-note">{response.result.feedback}</p>
                  {metrics.length === 0 ? (
                    <p className="panel-note compact">No hay métricas numéricas fiables para esta vista.</p>
                  ) : (
                    metrics.slice(0, 6).map(([key, value]) => (
                      <div className="data-row" key={key}>
                        <span>{labelMetric(key)}</span>
                        <strong>{formatMetric(value, key)}</strong>
                      </div>
                    ))
                  )}
                </section>
              )}
            </div>
          </>
        )}
      </div>
    </section>
  );
}

function PrivacyPanel({ apiBase, setApiBase, isDev }: { apiBase: string; setApiBase: (value: string) => void; isDev: boolean }) {
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
            <p className="eyebrow">{isDev ? "Conexión" : "Dudas frecuentes"}</p>
            <h2>{isDev ? "API local" : "Qué significan los datos"}</h2>
          </div>
        </div>
        {isDev ? (
          <>
            <label className="settings-label">
              URL de la aplicación local
              <div className="input-wrap">
                <Server size={17} />
                <input value={apiBase} onChange={(event) => setApiBase(normalizeApiBase(event.target.value))} />
              </div>
            </label>
            <div className="deployment-note">
              <FileImage size={20} />
              <p>
                Usa <code>http://localhost:8000</code> en el mismo equipo o <code>http://IP_DEL_HOST:8000</code> desde otro dispositivo de la misma red.
              </p>
            </div>
          </>
        ) : (
          <div className="privacy-steps single">
            <PrivacyStep title="Estado global" text="Resume la peor señal detectada en la captura. Riesgo no es diagnóstico médico; indica que conviene ajustar postura o puesto." />
            <PrivacyStep title="Métricas" text="Son medidas geométricas extraídas de puntos corporales visibles. Si faltan puntos, la captura puede aparecer como datos insuficientes." />
            <PrivacyStep title="Historial" text="Permite revisar cada captura y entender qué componente provocó el aviso, sin guardar la imagen original." />
            <PrivacyStep title="Recomendaciones" text="Se generan a partir de patrones repetidos, no de una captura aislada. Cuantas más capturas haya, más útiles serán." />
          </div>
        )}
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

function AnalysisDetail({ record, onClose }: { record: ReviewRecord; onClose: () => void }) {
  const metrics = Object.entries(record.metrics ?? {}).filter(([, value]) => value !== null);
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);
  return (
    <div className="detail-backdrop" role="presentation" onClick={onClose}>
      <section className="detail-panel" role="dialog" aria-modal="true" aria-label="Detalle de captura" onClick={(event) => event.stopPropagation()}>
        <header className="detail-header">
          <div>
            <p className="eyebrow">{formatDateTime(record.createdAt)} · {viewCopy[record.view].label}</p>
            <h2>{record.fileName}</h2>
          </div>
          <button className="theme-button compact" type="button" onClick={onClose} aria-label="Cerrar detalle">
            <X size={17} />
          </button>
        </header>
        <ResultPanel result={record} metrics={metrics} />
      </section>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
