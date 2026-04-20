import cv2 # takes time to execute
import os
from inference import get_model
from dotenv import load_dotenv

# Load variables from .env into the environment
load_dotenv() # loads api key

def main():
    print("Loading model weights (requires internet on first run)...")
    # Initialize the specific model version you linked
    model = get_model(model_id="posture_correction_v4/1")

    # Open the default laptop webcam (ID 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("\n--- System Ready ---")
    print("Focus on the camera window.")
    print("[c] - Capture and classify current frame")
    print("[q] - Quit the application\n")

    while True:
        # Continuously read frames to keep the camera feed live
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab a frame from the camera.")
            break

        # Show the live feed to the user
        cv2.imshow('Ergonomic Posture Tester', frame)

        # Listen for keyboard inputs
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Exiting...")
            break
            
        elif key == ord('c'):
            print("\n--- Processing Capture ---")
            # Run the local inference engine on the current frame
            results = model.infer(frame)
            
            # The model returns a list of results (one per image passed). 
            # We passed one frame, so we access index 0.
            predictions = results[0].predictions
            
            if not predictions:
                print("Result: No specific posture classes detected.")
            else:
                for pred in predictions:
                    # Output the class name and the model's confidence score
                    print(f"Result: {pred.class_name} | Confidence: {pred.confidence:.2f}")

    # Gracefully release hardware resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()