import cv2
import numpy as np 

print("--- STARTING CAMERA ACCESS DIAGNOSTIC (DirectShow) ---")
print("[Step 1] Attempting to acquire exclusive hardware control...")

# cv2.CAP_DSHOW forces OpenCV to use the Windows DirectShow API, 
# which handles hardware locks much more gracefully than the default backend (MSMF).
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

if not cap.isOpened():
    # Case 1: Driver-level lock. DirectShow could not even initialize.
    print(">> RESULT: [FAILURE] Could not open connection.")
    print(">> STATUS: The camera is strictly locked by another application or disconnected.")
else:
    print("[Step 2] Connection established. Requesting a test frame from Windows Frame Server...")
    
    # We attempt to read a single frame to verify if there is an actual video stream.
    ret, frame = cap.read()
    
    if not ret or frame is None or frame.size == 0:
        # Case 2: False Positive Opening. The camera "opened" but the data buffer is empty.
        print(">> RESULT: [FAILURE] Connection open, but the data buffer is empty.")
        print(">> STATUS: Stream is intercepted (e.g., by WhatsApp/Teams). Triggering Graceful Fallback.")
    else:
        print(f"[Step 3] Frame successfully received. Matrix dimensions: {frame.shape}")
        print("[Step 4] Starting image integrity analysis (Dummy Frame Detection)...")
        
        # --- NOISE-TOLERANT VALIDATION THRESHOLD ---
        # We calculate the mean intensity of all pixels. 
        # Values range from 0.0 (Absolute Black) to 255.0 (Absolute White).
        average_intensity = np.mean(frame)
        print(f"   -> Calculated mean intensity: {average_intensity:.4f} / 255.0")
        
        # We use 1.0 as the threshold. If it is lower, we assume it is a black frame
        # delivered by Windows for protection, ignoring residual digital noise.
        is_black = average_intensity < 1.0
        
        if is_black:
            # Case 3: Dummy Frame. We have a matrix of the correct size, but it is filled with dark pixels/noise.
            print(">> RESULT: [FAILURE] The frame is visually black (OS Dummy Frame).")
            print(">> STATUS: Camera is in use by another app. Triggering Graceful Fallback.")
        else:
            # Case 4: Total Success. We have real access to the sensor data.
            print(">> RESULT: [SUCCESS] The frame contains valid visual data.")
            print(">> STATUS: Camera is free. Ready to send data to the AI model.")
        
        # We save the image for manual verification, regardless of whether it is valid or not.
        filename = "cameraAccess_test_image.jpg"
        cv2.imwrite(filename, frame) 
        print(f"[INFO] A copy of the captured frame has been saved as '{filename}' in this directory.")
        
    # It is crucial to release the resource immediately in diagnostic scripts.
    cap.release()
    print("--- DIAGNOSTIC FINISHED. RESOURCES RELEASED ---")