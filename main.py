# Import the necessary modules
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
import cv2
import numpy as np


# Custom function to handle predictions and print detected objects
def custom_on_prediction(predictions, frame):
    try:
        # print(f"Predictions: {predictions}")

        for prediction in predictions['predictions']:
            print(f"Prediction: {prediction['class']} and confidence: {prediction['confidence']}")

        
        # Render the bounding boxes on the frame
        render_boxes(predictions, frame)
        
        # Display the frame with predictions
        # TASKS:
        # 1. design a bot which will say the distance of the object. use arduino. it should just say the specific object is closing or moving far. must not repeat again and again.
        
        if frame is not None and isinstance(frame, np.ndarray):
            cv2.imshow('Predictions', frame)
        else:
            print("[!] Invalid frame detected!")
        
        # Add a delay to allow the frame to be displayed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()
        
    except Exception as e:
        print(f"Error in custom_on_prediction: {e}")

# Initialize a pipeline object
try:
    pipeline = InferencePipeline.init(

        # sample models ['furniture-ngpea/1']
        model_id="living-room-items/1",  # Replace with your Roboflow model ID
        video_reference=0,  # 0 to use the built-in webcam
        api_key="G0HdUVNRRR2g5eGLP20Z",  # Your Roboflow API key
        on_prediction=custom_on_prediction,  # Use the custom function for predictions
        
    )

    pipeline.start()
    pipeline.join()
except Exception as e:
    print(f"Error initializing pipeline: {e}")
finally:
    cv2.destroyAllWindows()  # Ensure all OpenCV windows are destroyed