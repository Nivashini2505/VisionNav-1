# Import the necessary modules
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
import cv2

# Initialize a pipeline object
pipeline = InferencePipeline.init(
    model_id="critical-objects/1",  # Replace with your Roboflow model ID
    video_reference=1,  # 0 to use the built-in webcam
    api_key="G0HdUVNRRR2g5eGLP20Z",  # Your Roboflow API key
    on_prediction=render_boxes,  # Function to run after each prediction for visualization
)

# Function to check for 'q' key press
def check_exit_key():
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return True
    return False

# Start the pipeline
pipeline.start()

try:
    while True:
        # Continuously process the video feed
        if check_exit_key():
            print("Exiting...")
            break
finally:
    pipeline.stop()  # Stop the pipeline when exiting
    pipeline.join()  # Ensure the pipeline is properly closed
