from __future__ import annotations

import argparse
import ctypes
import json
import os
import site
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YOLO_WEIGHTS = PROJECT_ROOT / "models" / "yolo" / "yolov8s-pose.pt"


MEDIAPIPE_REQUIRED_IDS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_hip": 23,
    "right_hip": 24,
}

COCO_REQUIRED_IDS = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_hip": 11,
    "right_hip": 12,
}


def progress(iterable, **kwargs):
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


def empty_pose_metrics(model_name: str, runtime_ms: float, backend: str) -> dict:
    return {
        "model": model_name,
        "backend": backend,
        "runtime_ms": runtime_ms,
        "total_keypoints": 0,
        "confident_keypoints": 0,
        "required_keypoints_present": 0,
        "required_keypoint_rate": 0.0,
        "neck_ready": False,
        "trunk_ready": False,
        "left_elbow_ready": False,
        "right_elbow_ready": False,
        "upper_limb_ready": False,
        "upper_body_ready": False,
        "full_ergonomic_ready": False,
        "ergonomic_ready": False,
        "can_measure_core_angles": False,
    }


def compute_pose_support(required_scores: dict[str, float], threshold: float = 0.3) -> dict:
    normalized_scores = {
        name: 0.0 if score is None else float(score)
        for name, score in required_scores.items()
    }

    def has(*names: str) -> bool:
        return all(normalized_scores.get(name, 0.0) >= threshold for name in names)

    required_keypoints_present = sum(score >= threshold for score in normalized_scores.values())
    total_required = len(normalized_scores)
    neck_ready = has("nose", "left_shoulder", "right_shoulder")
    trunk_ready = has("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    left_elbow_ready = has("left_shoulder", "left_elbow")
    right_elbow_ready = has("right_shoulder", "right_elbow")
    upper_limb_ready = left_elbow_ready or right_elbow_ready
    upper_body_ready = neck_ready and upper_limb_ready
    full_ergonomic_ready = upper_body_ready and trunk_ready

    return {
        "required_keypoints_present": required_keypoints_present,
        "required_keypoint_rate": required_keypoints_present / total_required if total_required else 0.0,
        "neck_ready": neck_ready,
        "trunk_ready": trunk_ready,
        "left_elbow_ready": left_elbow_ready,
        "right_elbow_ready": right_elbow_ready,
        "upper_limb_ready": upper_limb_ready,
        "upper_body_ready": upper_body_ready,
        "full_ergonomic_ready": full_ergonomic_ready,
        "ergonomic_ready": upper_body_ready,
        "can_measure_core_angles": upper_body_ready,
    }


def build_pose_metrics(
    model_name: str,
    scores: list[float],
    required_ids: dict[str, int],
    threshold: float,
    runtime_ms: float,
    backend: str,
) -> dict:
    if not scores:
        return empty_pose_metrics(model_name=model_name, runtime_ms=runtime_ms, backend=backend)

    required_scores = {
        name: float(scores[index]) if index < len(scores) else 0.0
        for name, index in required_ids.items()
    }

    return {
        "model": model_name,
        "backend": backend,
        "runtime_ms": runtime_ms,
        "total_keypoints": len(scores),
        "confident_keypoints": int(sum(float(score) >= threshold for score in scores)),
        **compute_pose_support(required_scores, threshold=threshold),
    }


def ensure_mediapipe_model(path: Path, url: str) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(path, "wb") as fh:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    return path


def preload_tensorflow_gpu_libs() -> None:
    nvidia_root = next((Path(path) / "nvidia" for path in site.getsitepackages() if (Path(path) / "nvidia").exists()), None)
    if nvidia_root is None:
        return

    candidates = [
        Path("/lib/x86_64-linux-gnu/libcuda.so.1"),
        nvidia_root / "cuda_runtime/lib/libcudart.so.12",
        nvidia_root / "nvjitlink/lib/libnvJitLink.so.12",
        nvidia_root / "cublas/lib/libcublasLt.so.12",
        nvidia_root / "cublas/lib/libcublas.so.12",
        nvidia_root / "cudnn/lib/libcudnn.so.9",
        nvidia_root / "cudnn/lib/libcudnn_graph.so.9",
        nvidia_root / "cufft/lib/libcufft.so.11",
        nvidia_root / "cusparse/lib/libcusparse.so.12",
        nvidia_root / "cusolver/lib/libcusolver.so.11",
        nvidia_root / "curand/lib/libcurand.so.10",
        nvidia_root / "nccl/lib/libnccl.so.2",
    ]
    pending = [candidate for candidate in candidates if candidate.exists()]
    for _ in range(4):
        next_pending = []
        progress_made = False
        for candidate in pending:
            try:
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                progress_made = True
            except OSError:
                next_pending.append(candidate)
        if not next_pending or not progress_made:
            break
        pending = next_pending


def run_yolo_batch(
    manifest: list[dict],
    weights: str,
    device: str,
    warmup_images: int,
    min_confidence: float,
) -> list[dict]:
    import numpy as np
    import torch
    from ultralytics import YOLO

    runtime_device: str | int
    if device == "auto":
        runtime_device = 0 if torch.cuda.is_available() else "cpu"
    elif device.isdigit():
        runtime_device = int(device)
    else:
        runtime_device = device

    model = YOLO(weights)
    backend = f"torch_cuda:{runtime_device}" if runtime_device != "cpu" else "torch_cpu"

    def infer(image_path: Path) -> dict:
        start = time.perf_counter()
        result = model.predict(source=str(image_path), verbose=False, device=runtime_device)[0]
        runtime_ms = (time.perf_counter() - start) * 1000

        if result.keypoints is None:
            return empty_pose_metrics("yolo_pose", runtime_ms=runtime_ms, backend=backend)

        keypoint_data = getattr(result.keypoints, "data", None)
        if keypoint_data is None or len(keypoint_data) == 0:
            return empty_pose_metrics("yolo_pose", runtime_ms=runtime_ms, backend=backend)

        selected_index = 0
        if getattr(result, "boxes", None) is not None and getattr(result.boxes, "conf", None) is not None:
            selected_index = int(result.boxes.conf.argmax().item())

        arr = keypoint_data[selected_index].cpu().numpy()
        if arr.ndim != 2:
            arr = np.asarray(arr).reshape(-1, arr.shape[-1])

        scores = arr[:, 2].astype(float).tolist() if arr.shape[1] >= 3 else [1.0] * arr.shape[0]
        return build_pose_metrics(
            model_name="yolo_pose",
            scores=scores,
            required_ids=COCO_REQUIRED_IDS,
            threshold=min_confidence,
            runtime_ms=runtime_ms,
            backend=backend,
        )

    for item in manifest[:warmup_images]:
        infer(Path(item["image_path"]))

    results = []
    for item in progress(manifest, desc="YOLO Pose", leave=False):
        image_path = Path(item["image_path"])
        try:
            metrics = infer(image_path)
            metrics["image_name"] = image_path.name
            metrics["group"] = item.get("group")
            metrics["split"] = item.get("split")
            metrics["error"] = None
        except Exception as exc:
            metrics = {
                "model": "yolo_pose",
                "backend": backend,
                "image_name": image_path.name,
                "group": item.get("group"),
                "split": item.get("split"),
                "runtime_ms": None,
                "total_keypoints": None,
                "confident_keypoints": None,
                "required_keypoints_present": None,
                "required_keypoint_rate": None,
                "neck_ready": False,
                "trunk_ready": False,
                "left_elbow_ready": False,
                "right_elbow_ready": False,
                "upper_limb_ready": False,
                "upper_body_ready": False,
                "full_ergonomic_ready": False,
                "ergonomic_ready": False,
                "can_measure_core_angles": False,
                "error": str(exc),
            }
        results.append(metrics)
    return results


def run_movenet_batch(
    manifest: list[dict],
    warmup_images: int,
    input_size: int,
    min_confidence: float,
    device: str,
) -> list[dict]:
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device == "gpu":
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        os.environ.setdefault("TF_CUDNN_USE_FRONTEND", "0")
        preload_tensorflow_gpu_libs()

    import cv2
    import numpy as np
    import tensorflow as tf
    import tensorflow_hub as hub

    if device == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

    visible_gpus = tf.config.list_physical_devices("GPU")
    if device == "gpu" and not visible_gpus:
        raise RuntimeError("MoveNet GPU solicitado, pero TensorFlow no detecta ninguna GPU.")

    for gpu in visible_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4").signatures["serving_default"]
    backend = "tensorflow_gpu" if tf.config.list_physical_devices("GPU") else "tensorflow_cpu"

    def infer(image_path: Path) -> dict:
        image_bgr = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = tf.image.resize_with_pad(np.expand_dims(image_rgb, axis=0), input_size, input_size)
        input_tensor = tf.cast(resized, dtype=tf.int32)

        start = time.perf_counter()
        outputs = model(input_tensor)
        runtime_ms = (time.perf_counter() - start) * 1000

        keypoints = outputs["output_0"].numpy()[0, 0, :, :]
        scores = keypoints[:, 2].astype(float).tolist()
        return build_pose_metrics(
            model_name="movenet",
            scores=scores,
            required_ids=COCO_REQUIRED_IDS,
            threshold=min_confidence,
            runtime_ms=runtime_ms,
            backend=backend,
        )

    for item in manifest[:warmup_images]:
        infer(Path(item["image_path"]))

    results = []
    for item in progress(manifest, desc="MoveNet", leave=False):
        image_path = Path(item["image_path"])
        try:
            metrics = infer(image_path)
            metrics["image_name"] = image_path.name
            metrics["group"] = item.get("group")
            metrics["split"] = item.get("split")
            metrics["error"] = None
        except Exception as exc:
            metrics = {
                "model": "movenet",
                "backend": backend,
                "image_name": image_path.name,
                "group": item.get("group"),
                "split": item.get("split"),
                "runtime_ms": None,
                "total_keypoints": None,
                "confident_keypoints": None,
                "required_keypoints_present": None,
                "required_keypoint_rate": None,
                "neck_ready": False,
                "trunk_ready": False,
                "left_elbow_ready": False,
                "right_elbow_ready": False,
                "upper_limb_ready": False,
                "upper_body_ready": False,
                "full_ergonomic_ready": False,
                "ergonomic_ready": False,
                "can_measure_core_angles": False,
                "error": str(exc),
            }
        results.append(metrics)
    return results


def run_mediapipe_batch(
    manifest: list[dict],
    model_path: Path,
    model_url: str,
    warmup_images: int,
    min_visibility: float,
) -> list[dict]:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    ensure_mediapipe_model(model_path, model_url)
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    model = vision.PoseLandmarker.create_from_options(options)
    backend = "mediapipe_tasks"

    def infer(image_path: Path) -> dict:
        image_bgr = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        start = time.perf_counter()
        result = model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb))
        runtime_ms = (time.perf_counter() - start) * 1000

        if not result.pose_landmarks:
            return empty_pose_metrics("mediapipe_pose", runtime_ms=runtime_ms, backend=backend)

        landmarks = list(result.pose_landmarks[0])
        scores = []
        for landmark in landmarks:
            confidence = getattr(landmark, "visibility", None)
            if confidence is None:
                confidence = getattr(landmark, "presence", None)
            scores.append(0.0 if confidence is None else float(confidence))

        return build_pose_metrics(
            model_name="mediapipe_pose",
            scores=scores,
            required_ids=MEDIAPIPE_REQUIRED_IDS,
            threshold=min_visibility,
            runtime_ms=runtime_ms,
            backend=backend,
        )

    try:
        for item in manifest[:warmup_images]:
            infer(Path(item["image_path"]))

        results = []
        for item in progress(manifest, desc="MediaPipe Pose", leave=False):
            image_path = Path(item["image_path"])
            try:
                metrics = infer(image_path)
                metrics["image_name"] = image_path.name
                metrics["group"] = item.get("group")
                metrics["split"] = item.get("split")
                metrics["error"] = None
            except Exception as exc:
                metrics = {
                    "model": "mediapipe_pose",
                    "backend": backend,
                    "image_name": image_path.name,
                    "group": item.get("group"),
                    "split": item.get("split"),
                    "runtime_ms": None,
                    "total_keypoints": None,
                    "confident_keypoints": None,
                    "required_keypoints_present": None,
                    "required_keypoint_rate": None,
                    "neck_ready": False,
                    "trunk_ready": False,
                    "left_elbow_ready": False,
                    "right_elbow_ready": False,
                    "upper_limb_ready": False,
                    "upper_body_ready": False,
                    "full_ergonomic_ready": False,
                    "ergonomic_ready": False,
                    "can_measure_core_angles": False,
                    "error": str(exc),
                }
            results.append(metrics)
        return results
    finally:
        model.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["yolo_pose", "movenet", "mediapipe_pose"])
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--warmup-images", type=int, default=2)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--input-size", type=int, default=192)
    parser.add_argument("--yolo-weights", default=str(DEFAULT_YOLO_WEIGHTS))
    parser.add_argument("--yolo-device", default="auto")
    parser.add_argument("--movenet-device", default="cpu", choices=["auto", "gpu", "cpu"])
    parser.add_argument("--mediapipe-model-path", default="")
    parser.add_argument("--mediapipe-model-url", default="")
    args = parser.parse_args()

    manifest = json.loads(Path(args.input_json).read_text())
    if args.model == "yolo_pose":
        results = run_yolo_batch(
            manifest=manifest,
            weights=args.yolo_weights,
            device=args.yolo_device,
            warmup_images=args.warmup_images,
            min_confidence=args.min_confidence,
        )
    elif args.model == "movenet":
        results = run_movenet_batch(
            manifest=manifest,
            warmup_images=args.warmup_images,
            input_size=args.input_size,
            min_confidence=args.min_confidence,
            device=args.movenet_device,
        )
    else:
        results = run_mediapipe_batch(
            manifest=manifest,
            model_path=Path(args.mediapipe_model_path),
            model_url=args.mediapipe_model_url,
            warmup_images=args.warmup_images,
            min_visibility=args.min_confidence,
        )

    Path(args.output_json).write_text(json.dumps(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
