from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd

from .paths import MEDIAPIPE_TASK_MODEL_PATH

# --- DEFINICIÓN DE LA ANATOMÍA VIRTUAL ---

# Mapeo de nombres legibles a los índices oficiales de MediaPipe.
# Esto facilita la lectura.
LANDMARK_IDS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}

# Definición de las conexiones del esqueleto (huesos).
# Útil tanto para visualización como para cálculos de ángulos entre segmentos.
SKELETON_SEGMENTS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
]


def _mediapipe_imports():
    """
    Gestión de dependencias
    Solo importa MediaPipe cuando se necesita y lanza un error descriptivo si falta,
    evitando que el proyecto falle simplemente por importar el módulo.
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
    except ImportError as exc:
        raise ImportError(
            "Falta mediapipe. Instala la dependencia con `python -m pip install mediapipe` antes de ejecutar esta fase."
        ) from exc
    return mp, vision, BaseOptions


@dataclass(frozen=True)
class MediaPipePoseConfig:
    """
    Configuración de los umbrales de confianza del modelo
    Permite ajustar la sensibilidad del modelo para filtrar detecciones dudosas.
    """
    model_path: Path = MEDIAPIPE_TASK_MODEL_PATH
    min_pose_detection_confidence: float = 0.5 # Confianza para detectar una persona
    min_pose_presence_confidence: float = 0.5  # Confianza de que la pose sigue ahí
    min_tracking_confidence: float = 0.5       # Confianza para el seguimiento entre frames
    min_visibility: float = 0.3                # Umbral para considerar un punto como 'visible'


class MediaPipePoseEstimator:
    """
    Gestor del modelo de IA MediaPipe.
    Esta clase controla el 'ciclo de vida' del detector de posturas. 
    Garantiza que solo consuma memoria cuando se está usando y 
    se apague automáticamente al terminar, evitando que el ordenador 
    del trabajador se vuelva lento o se bloquee por falta de RAM.
    """
    def __init__(self, config: MediaPipePoseConfig | None = None):
        self.config = config or MediaPipePoseConfig()
        self._model = None
        self._mp = None

    def __enter__(self) -> "MediaPipePoseEstimator":
        """
        Inicializa el modelo PoseLandmarker de MediaPipe con las opciones configuradas.
        """
        mp, vision, BaseOptions = _mediapipe_imports()
        
        # Validación de seguridad: ¿está el archivo del modelo en su sitio?
        if not self.config.model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo de MediaPipe en {self.config.model_path}."
            )

        # Configuración técnica del motor de visión
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.config.model_path)),
            running_mode=vision.RunningMode.IMAGE, # Optimizado para procesamiento foto a foto
            num_poses=1,                           # Solo buscamos a un trabajador por imagen
            min_pose_detection_confidence=self.config.min_pose_detection_confidence,
            min_pose_presence_confidence=self.config.min_pose_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            output_segmentation_masks=False,       # Desactivado para ahorrar CPU (no necesitamos siluetas)
        )
        
        self._mp = mp
        self._model = vision.PoseLandmarker.create_from_options(options)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """
        Se asegura de cerrar el modelo de MediaPipe y liberar la memoria RAM
        al terminar el proceso, incluso si el programa ha tenido un error.
        """
        if self._model is not None:
            self._model.close()
        self._model = None

    def infer_image(self, image_path: str | Path, metadata: dict | None = None) -> dict:
        """
        Carga una imagen, la pasa por el modelo y devuelve un diccionario con 
        las coordenadas (x, y, z) de cada parte del cuerpo detectada.
        """
        # Validación de seguridad: no podemos ver si no hemos abierto los ojos
        if self._model is None or self._mp is None:
            raise RuntimeError("El estimador de MediaPipe no esta inicializado.")

        image_path = Path(image_path)
        # Lectura de imagen: OpenCV lee en formato BGR por defecto
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")

        # MediaPipe requiere formato RGB, por lo que convertimos los colores
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Aquí es donde la red neuronal detecta el cuerpo
        result = self._model.detect(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image_rgb)
        )

        # Preparación de la fila
        row = {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "group": metadata.get("group") if metadata else None,
            "split": metadata.get("split") if metadata else None,
            "image_width": int(image_rgb.shape[1]),
            "image_height": int(image_rgb.shape[0]),
            "pose_detected": bool(result.pose_landmarks),
            "pose_landmarks_count": int(len(result.pose_landmarks[0])) if result.pose_landmarks else 0,
            "visible_landmarks_count": 0,
        }

        # Inicializamos todas las columnas de coordenadas a 'None' por si falla
        for landmark_name in LANDMARK_IDS:
            row[f"{landmark_name}_x"] = None
            row[f"{landmark_name}_y"] = None
            row[f"{landmark_name}_z"] = None
            row[f"{landmark_name}_visibility"] = None

        if not result.pose_landmarks:
            return row

        # EXTRACCIÓN DE PUNTOS CLAVE
        # MediaPipe devuelve coordenadas normalizadas (0.0 a 1.0)
        landmarks = result.pose_landmarks[0]
        visible_count = 0
        for landmark_name, landmark_id in LANDMARK_IDS.items():
            landmark = landmarks[landmark_id]
            visibility = float(getattr(landmark, "visibility", 0.0))
            
            row[f"{landmark_name}_x"] = float(landmark.x)
            row[f"{landmark_name}_y"] = float(landmark.y)
            row[f"{landmark_name}_z"] = float(landmark.z) # Profundidad relativa
            row[f"{landmark_name}_visibility"] = visibility
            
            # Solo contamos el punto si está suficientemente seguro
            if visibility >= self.config.min_visibility:
                visible_count += 1

        row["visible_landmarks_count"] = visible_count
        return row


def run_mediapipe_pose_batch(
    sample_df: pd.DataFrame,
    config: MediaPipePoseConfig | None = None,
) -> pd.DataFrame:
    """
    Procesa un listado entero de imágenes una tras otra de forma eficiente,
    devolviendo una tabla de Pandas con todos los resultados.
    """
    if sample_df.empty:
        return pd.DataFrame()

    rows = []
    # Usamos el Context Manager (with) para abrir y cerrar el modelo automáticamente
    with MediaPipePoseEstimator(config=config) as estimator:
        for item in sample_df.to_dict(orient="records"):
            # Analizamos cada imagen y guardamos el diccionario resultante
            rows.append(
                estimator.infer_image(
                    item["image_path"],
                    metadata={"group": item.get("group"), "split": item.get("split")},
                )
            )
            
    # Convertimos la lista de resultados en un DataFrame final
    return pd.DataFrame(rows)