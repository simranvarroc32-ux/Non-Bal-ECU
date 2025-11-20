import cv2
import os
import time
import tkinter as tk  # Only needed for the test UI

# CONFIGURATION
SAVE_FOLDER = "inspection_images"  # Folder where images will be saved
CAMERA_INDEX = 0  # 0 is usually the default USB camera

# Ensure the folder exists
os.makedirs(SAVE_FOLDER, exist_ok=True)

def capture_image():
    """
    Triggers the camera to take one photo and saves it to the folder.
    Call this function from your UI button.
    """
    # Open connection to camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Allow camera to warm up (optional, but often helps with brightness)
    time.sleep(0.1) 
    
    ret, frame = cap.read()
    
    if ret:
        # Generate a unique filename based on time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"img_{timestamp}.jpg"
        filepath = os.path.join(SAVE_FOLDER, filename)
        
        # Save the image
        cv2.imwrite(filepath, frame)
        print(f"SUCCESS: Image saved to {filepath}")
    else:
        print("Error: Could not read frame.")

    # Release the camera so other apps can use it
    cap.release()

# --- BELOW IS JUST FOR TESTING (SIMULATING YOUR UI) ---
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Camera Trigger Test")
    root.geometry("300x150")

    btn = tk.Button(root, text="CAPTURE IMAGE", font=("Arial", 14), 
                    command=capture_image, bg="green", fg="white")
    btn.pack(expand=True)

    root.mainloop()