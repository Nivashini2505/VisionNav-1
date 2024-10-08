# Import necessary modules
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
import cv2
import numpy as np


# Custom function to handle predictions and print detected objects
def custom_on_prediction(predictions, frame):
    try:
        for prediction in predictions['predictions']:
            object_class = prediction['class']
            print(f"Object {object_class}")

        # Render the bounding boxes on the frame
        render_boxes(predictions, frame)

        # Check if the frame is valid
        if frame is not None and isinstance(frame, np.ndarray):
            # Display the frame with predictions
            cv2.imshow('Predictions', frame)
        else:
            print("Invalid frame detected, skipping display.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()

    except Exception as e:
        print(f"Error in custom_on_prediction: {e}")

# Initialize a pipeline object
try:
    pipeline = InferencePipeline.init(
        model_id="living-room-items/1",
        video_reference=1,
        api_key="G0HdUVNRRR2g5eGLP20Z",
        on_prediction=custom_on_prediction,
    )

    pipeline.start()
    pipeline.join()
except Exception as e:
    print(f"Error initializing pipeline: {e}")
finally:
    cv2.destroyAllWindows()
