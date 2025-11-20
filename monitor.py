import time
import sys
import os
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# CONFIGURATION
WATCH_FOLDER = "inspection_images"
TRAIN_SCRIPT = "train.py"  # The script used to test OK/NG

class NewImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        # Ignore if a folder is created, we only want files
        if event.is_directory:
            return

        # Only react to image files (you can add png, bmp etc)
        if event.src_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"\n[EVENT] New image detected: {event.src_path}")
            
            # Wait briefly to ensure file write is complete
            time.sleep(0.5) 
            
            # Run the training/testing script
            self.run_test_script(event.src_path)

    def run_test_script(self, image_path):
        print(f" -> Running {TRAIN_SCRIPT} on {image_path}...")
        
        try:
            # This calls: python train.py <path_to_image>
            # We use sys.executable to ensure we use the same python interpreter
            result = subprocess.run(
                [sys.executable, TRAIN_SCRIPT, image_path],
                capture_output=True,
                text=True
            )
            
            # Print the output from train.py
            print(" -> TEST RESULT OUTPUT:")
            print(result.stdout)
            
            if result.stderr:
                print(" -> ERRORS:", result.stderr)
                
        except Exception as e:
            print(f"Error running script: {e}")

if __name__ == "__main__":
    # Ensure the folder exists before watching it
    os.makedirs(WATCH_FOLDER, exist_ok=True)

    event_handler = NewImageHandler()
    observer = Observer()
    observer.schedule(event_handler, path=WATCH_FOLDER, recursive=False)
    
    print(f"Monitoring folder: '{WATCH_FOLDER}' for new images...")
    print("Press Ctrl+C to stop.")
    
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()