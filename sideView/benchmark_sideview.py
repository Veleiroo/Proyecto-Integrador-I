# The execution of this script will give results similar to the following:
# --- Benchmark Results ---
#     Model  Avg Latency (ms)  Detection Rate (%)  Avg Visible Keypoints
# MediaPipe        217.869530               100.0                    6.2
#    YOLOv8        597.611995               100.0                    7.0
#   MoveNet        507.443370               100.0                    5.6

# The visible keypoints average is over the 7 considered keypoints.
# The detection rate is the percentage of images in which a person was detected.

# Based on the results above, Movenet is discarded. Between mediapipe and yolo we choose
# mediapipe because it is good enough, because it is faster and because it is also used
# in the front view.

import os
import json
import time
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Constants and Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "datasets" / "ikornproject_sitting-posture-rofqf_v4"
YOLO_WEIGHTS = PROJECT_ROOT / "models" / "yolo" / "yolov8s-pose.pt"
MEDIAPIPE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"

# Landmark Definitions
COCO_IDS = {
    "nose": 0, "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8, "left_hip": 11, "right_hip": 12
}
MP_IDS = {
    "nose": 0, "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14, "left_hip": 23, "right_hip": 24
}

SKELETON = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip")
]

class BaseEstimator:
    def __init__(self, name):
        self.name = name
    def infer(self, image_path):
        raise NotImplementedError
    def close(self):
        pass

class MediaPipeEstimator(BaseEstimator):
    def __init__(self):
        super().__init__("MediaPipe")
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        import urllib.request

        model_path = PROJECT_ROOT / "models" / "mediapipe" / "pose_landmarker_heavy.task"
        if not model_path.exists():
            print(f"Downloading MediaPipe model to {model_path}...")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, model_path)

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.mp = mp

    def infer(self, image_path):
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=img_rgb)
        
        start = time.perf_counter()
        result = self.detector.detect(mp_image)
        latency = (time.perf_counter() - start) * 1000

        landmarks_dict = {}
        if result.pose_landmarks:
            pts = result.pose_landmarks[0]
            for name, idx in MP_IDS.items():
                lm = pts[idx]
                landmarks_dict[name] = {"x": lm.x, "y": lm.y, "visibility": lm.visibility}
        
        return {"landmarks": landmarks_dict, "latency": latency}

    def close(self):
        self.detector.close()

class YOLOEstimator(BaseEstimator):
    def __init__(self):
        super().__init__("YOLOv8")
        from ultralytics import YOLO
        YOLO_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(YOLO_WEIGHTS)) # Use the absolute path variable

    def infer(self, image_path):
        start = time.perf_counter()
        results = self.model.predict(str(image_path), verbose=False)
        latency = (time.perf_counter() - start) * 1000
        
        landmarks_dict = {}
        if len(results) > 0 and results[0].keypoints is not None:
            # Get first person detected
            kpts = results[0].keypoints.data[0].cpu().numpy()
            for name, idx in COCO_IDS.items():
                if idx < len(kpts):
                    pt = kpts[idx]
                    landmarks_dict[name] = {"x": pt[0]/results[0].orig_shape[1], "y": pt[1]/results[0].orig_shape[0], "visibility": pt[2] if len(pt) > 2 else 1.0}
        
        return {"landmarks": landmarks_dict, "latency": latency}

class MoveNetEstimator(BaseEstimator):
    def __init__(self):
        super().__init__("MoveNet")
        import tensorflow as tf
        import tensorflow_hub as hub
        self.model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4").signatures["serving_default"]
        self.tf = tf

    def infer(self, image_path):
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_size = 192
        input_image = cv2.resize(img_rgb, (input_size, input_size))
        input_image = np.expand_dims(input_image, axis=0)
        input_image = self.tf.cast(input_image, dtype=self.tf.int32)

        start = time.perf_counter()
        outputs = self.model(input_image)
        latency = (time.perf_counter() - start) * 1000

        keypoints = outputs["output_0"].numpy()[0, 0, :, :] # [17, 3] -> y, x, score
        landmarks_dict = {}
        for name, idx in COCO_IDS.items():
            pt = keypoints[idx]
            landmarks_dict[name] = {"x": pt[1], "y": pt[0], "visibility": pt[2]}
        
        return {"landmarks": landmarks_dict, "latency": latency}

def calculate_metrics(results_list):
    df = pd.DataFrame(results_list)
    summary = {
        "Model": df["model"].iloc[0],
        "Avg Latency (ms)": df["latency"].mean(),
        "Detection Rate (%)": (df["detected"].sum() / len(df)) * 100,
        "Avg Visible Keypoints": df["visible_count"].mean()
    }
    return summary

def draw_skeleton(ax, landmarks, title):
    img_path = landmarks["image_path"]
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    ax.imshow(img)
    
    pts = landmarks["landmarks"]
    for name, pt in pts.items():
        if pt["visibility"] > 0.3:
            ax.scatter(pt["x"] * w, pt["y"] * h, s=20, color='red')
            
    for start, end in SKELETON:
        if start in pts and end in pts:
            p1 = pts[start]
            p2 = pts[end]
            if p1["visibility"] > 0.3 and p2["visibility"] > 0.3:
                ax.plot([p1["x"] * w, p2["x"] * w], [p1["y"] * h, p2["y"] * h], color='yellow', linewidth=1)
    
    ax.set_title(f"{title}\n{len(pts)} pts", fontsize=8)
    ax.axis('off')

def main():
    print("Discovering images in ikorn dataset...")
    image_paths = list(DATASET_PATH.glob("**/*.jpg"))[:20] # Take a sample of 20
    if not image_paths:
        print(f"No images found in {DATASET_PATH}")
        return

    estimators = [MediaPipeEstimator(), YOLOEstimator(), MoveNetEstimator()]
    all_metrics = []
    gallery_data = []

    for est in estimators:
        print(f"Running benchmark for {est.name}...")
        results = []
        for i, img_path in enumerate(tqdm(image_paths)):
            res = est.infer(img_path)
            res["model"] = est.name
            res["image_path"] = img_path
            res["detected"] = len(res["landmarks"]) > 0
            res["visible_count"] = sum(1 for p in res["landmarks"].values() if p["visibility"] > 0.3)
            results.append(res)
            if i < 3: # Save first 3 for gallery
                gallery_data.append(res)
        
        all_metrics.append(calculate_metrics(results))
        est.close()

    # Reporting
    metrics_df = pd.DataFrame(all_metrics)
    print("\n--- Benchmark Results ---")
    print(metrics_df.to_string(index=False))

    # Visualization
    num_imgs = 3
    num_models = len(estimators)
    fig, axes = plt.subplots(num_imgs, num_models, figsize=(num_models * 3, num_imgs * 3))
    
    for i in range(num_imgs):
        for j in range(num_models):
            # gallery_data is populated as [est1_img1, est1_img2, est1_img3, est2_img1, ...]
            # Each estimator has num_imgs items in gallery_data
            data = gallery_data[j * num_imgs + i] 
            draw_skeleton(axes[i, j], data, data["model"])

    plt.tight_layout()
    report_path = PROJECT_ROOT / "sideView" / "sideview_benchmark_report.png"
    plt.savefig(report_path)
    print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    main()
