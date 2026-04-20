# Ergonomic Posture Monitor MVP: System Specification

## 1. Project Overview
Office workers frequently suffer from poor ergonomic posture. This Minimum Viable Product (MVP) is an offline, local-first computer vision system designed to monitor and correct these habits. The system periodically captures user images at a low-frequency interval, extracts structural keypoints, evaluates postural health, and delivers non-intrusive feedback to the user.

---

## 2. System Architecture & Modules

### 2.1. Image Capture (Data Ingestion)
The system requires an image feed to operate. **In production, the system will actively capture images using an attached camera.** However, the architecture will retain the ability to ingest local image files to facilitate rapid testing and development. 

**Capture Rate (Sampling Frequency):**
To prevent computational bottlenecks and ensure smooth system performance without interrupting the user's workflow, the MVP will utilize a low-frequency sampling rate of **one image every 20 seconds**. 

To optimize the keypoint extraction model, the development team will evaluate three camera topologies. The final design will be selected based on the best balance of simplicity and feature extraction performance:
* **Option 1 (Front View):** Utilizes a standard laptop webcam. Only frontal images will be processed.
* **Option 2 (Side View):** Utilizes a profile/sagittal camera angle. Only side-view images will be processed.
* **Option 3 (Stereo View):** Simultaneous capture using both front and side cameras.

### 2.2. Keypoint Detection (Pose Estimation)
The system will utilize an AI model to extract spatial coordinates of critical anatomical joints. The team will benchmark the following models to determine the optimal balance between high joint-recognition accuracy and low computational overhead:
* **YOLO Pose**
* **MoveNet**
* **MediaPipe Pose**

Only some keypoints will be considered.
> **⚠️ ACTION REQUIRED: Keypoint selection**
> We must choose which keypoints to use. Optional: what joints those keypoints can be used for.


### 2.3. Geometric Calculation (Feature Engineering)
*Note: This module is currently provisional but strongly recommended.*
Instead of evaluating raw pixel coordinates, this module translates keypoints into joint angles. By calculating the geometric angles between connected joints, the system establishes scale-invariant data, which simplifies the downstream Posture Evaluation module.

### 2.4. Posture Evaluation (Inference)
This module analyzes the extracted keypoints or calculated angles to classify the user's posture. To avoid UI flickering and annoying the user with micro-corrections (e.g., leaning forward to grab a pen), the system utilizes **temporal smoothing**: a posture anomaly must be detected across **multiple consecutive frames** (e.g., 3 frames, totaling ~60 seconds of sustained bad posture) before it is flagged.

The team strictly prefers the simplest viable approach. If deterministic, hardcoded geometric rules (heuristics) are sufficient, they will be prioritized over training a dedicated Machine Learning model.

**Decoupled Evaluation Strategy (Deferred Decision):**
Because the specific ergonomic taxonomy (e.g., Slouching vs. Forward Head) and the risk assessment methodology will be determined empirically, this module must be highly decoupled.
* **Configuration-Driven:** Hardcoded rules, angle thresholds, and issue classifications must be stored in an external configuration file (e.g., `config.yaml`), allowing the team to tweak the logic without altering the core Python application.
* **Generic Payload:** The module will output a standardized dictionary/JSON payload to the Notification module. 
* **Data Collection Mode:** During early development, the module will support a "shadow mode" where it simply logs calculated joint angles to a local CSV file without triggering notifications, allowing the team to analyze real user data to define the final posture classes.

### 2.5. Notification (Actuation)
If an issue is confirmed, the system triggers an alert informing the worker of their posture classification. 

**Notification Design:** To maximize simplicity and reduce development overhead, the system will leverage native Operating System notifications (e.g., Windows toast notifications). This avoids the need for a custom graphical user interface while remaining discrete.

---

## 3. Non-Functional Requirements & Execution

* **Technology Stack:** The core system will be developed in **Python 3.10+**. Environment management and dependency isolation will be handled via a standard `.venv` (Python Virtual Environment).
* **Execution Lifecycle:** The application will operate as a background command-line interface (CLI) process. The user must keep the terminal window open during execution and can terminate the process via standard keyboard interrupts (`Ctrl+C`).
* **Privacy First:** All images captured via the camera must be immediately deleted from memory after keypoint extraction. Image retention is strictly limited to active development/debugging modes.
* **Offline Operation:** The pipeline must execute entirely locally without relying on external cloud APIs.
* **Portability:** The MVP will be published on GitHub. It must be highly portable, easy to download, install, and run. 
* **Resource Contention & Error Logging:** Windows typically restricts webcam access to a single application. If the camera is locked by another process (e.g., a Zoom or Teams video call), the system must not crash. Instead, it must gracefully fail, log the error to a local `app.log` file, skip the current evaluation cycle, and attempt capture again at the next interval.
* **Hardware Acceleration (Windows/GPU Integration):** * The target operating system for execution is **Windows**.
    * The system will be engineered to utilize GPU acceleration whenever possible. 
    * The specific GPU hardware configuration will be passed to the system as a **startup parameter**. 
    * The system will evaluate this parameter; if the hardware is supported and the environment is ready, it will execute inference on the GPU. If the hardware is unsupported or errors occur, the system must gracefully **fallback to CPU inference**. *(Note: For the MVP phase, this hardware logic is decoupled from the specific CV model selected).*
* **Manual Control (Pause/Resume):** To provide user agency and privacy, the system will support a "pause" state. 
    * The user can toggle monitoring by pressing a specific key (e.g., `P`) in the terminal.
    * While paused, the system will skip the Image Capture and Posture Evaluation cycles.
    * The AI model will remain loaded in memory to ensure that resuming the service is instantaneous, avoiding the latency of re-initializing the GPU or weights.
    * The system should trigger a native OS notification confirming the state change (e.g., "Posture Monitor Paused").
    * The temporal smoothing buffer shoould be cleared so when resumed the system has an empty buffer.

---

## 4. Validation & Datasets

System validation will rigorously test the accuracy of the Posture Issue Classification. Because classification relies entirely on upstream data, the Keypoint Detection accuracy will also be heavily audited. Validation will utilize both team-created proprietary images and publicly available tagged datasets.

### Current Data Sources
The team currently has access to the following public datasets:
* [Roboflow: sitting-posture-rofqf](https://universe.roboflow.com/ikornproject/sitting-posture-rofqf) *(Raw Images)*
This dataset has classes Good and Bad. Images are side view. 273 images. Not so interesting.

* [Roboflow: sitting-posture-ezkda](https://universe.roboflow.com/dataset-sqm0h/sitting-posture-ezkda) *(Raw Images)*
This dataset has classes backbadposture forwardbadposture and goodposture. Images are sideview but with slightly different angels. 938 images. Not so interesting.

* [Zenodo: Record 14230872](https://zenodo.org/records/14230872) *(Pre-extracted Coordinates)*
About this dataset:
This dataset contains skeletal pose data extracted from video recordings of 13 participants performing various sitting postures in home environments. The data was processed using MediaPipe Pose Heavy model and includes 4,800 frames of 3D skeletal coordinates (x, y, z) for 11 key body joints, with each frame manually labeled for both upper and lower body posture classifications.

The data is stored in CSV format with normalized coordinates relative to hip center, containing 33 input dimensions (11 joints × 3 coordinates) representing key skeletal points. To protect participant privacy, only the processed skeletal coordinates are included, with no raw video or image data due to privacy constraints.

Upper Body Labels:

    TUP: Upright trunk position
    TLB: Trunk leaning backward
    TLF: Trunk leaning forward
    TLR: Trunk leaning right
    TLL: Trunk leaning left

Lower Body Labels:

    LAP: Legs apart
    LWA: Legs wide apart
    LCS: Legs closed
    LCR: Legs crossed right over left
    LCL: Legs crossed left over right
    LLR: Legs lateral right
    LLL: Legs lateral left

Each frame in the dataset has been manually labeled and validated by experts, making it particularly suitable for developing and evaluating machine learning models for ergonomic monitoring systems, ambient assisted living applications, and general posture recognition research.

* [Roboflow: posture_correction_v4](https://universe.roboflow.com/posturecorrection/posture_correction_v4/browse) *(Images)*
Classes: "looks good" "sit up straight" "straighten head" and "Unlabeled".
Front view images of several people with slightly different angles.
4666 images.
Most interesting dataset.

## 5. Repository Configuration & Environment Management

To ensure portability, reproducibility, and ease of setup across different hardware configurations, the repository must be structured according to standard Python engineering practices.

* **Dependency Management:** All required external libraries (e.g., `inference`, `opencv-python`) must be explicitly declared and version-pinned in a `requirements.txt` file. This allows developers to easily reconstruct the execution environment using a standard package manager command (`pip install -r requirements.txt`).
* **Virtual Environment (`.venv`):** The project mandates an isolated Python Virtual Environment (`.venv`) for execution. The repository will not track the environment itself, only the instructions to recreate it, ensuring dependency isolation and preventing conflicts with the host operating system.
* **Version Control Exclusions (`.gitignore`):** To maintain a clean, secure, and lightweight repository, a `.gitignore` file must be implemented. It must strictly exclude:
    * The `.venv/` directory and `__pycache__/` files.
    * `.env` files containing private credentials (e.g., the Roboflow API key required for the initial weights download).
    * Local image captures, camera test outputs, or user-specific data.
    * Local execution logs (e.g., `app.log`).
* **Environment Variables (`.env`):** To adhere to security best practices, sensitive credentials such as API keys must never be hardcoded into the Python scripts. The system will rely on a local `.env` file to inject these variables into the runtime environment.
