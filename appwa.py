# Import necessary modules
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
import cv2
import serial
import numpy as np
import time

# Setup serial communication (adjust the port and baud rate as needed)
arduino_serial = serial.Serial('COM4', 9600, timeout=1)

# Function to read the distance from the Arduino
def read_distance():
    try:
        # Read a line of data from the Arduino
        distance_data = arduino_serial.readline().decode().strip()

        # Check if the data is valid and convert it to an integer
        if distance_data.isdigit():
            distance_mm = int(distance_data)
            distance_ft = distance_mm * 0.00328084  # Convert to feet
            return distance_ft
        else:
            return None
    except Exception as e:
        print(f"Error reading distance: {e}")
        return None

# Custom function to handle predictions and print detected objects
def custom_on_prediction(predictions, frame):
    try:
        # Read distance from Arduino
        distance_ft = read_distance()

        if distance_ft is not None:
            for prediction in predictions['predictions']:
                object_class = prediction['class']
                print(f"Object '{object_class}' is at a distance of {distance_ft:.2f} feet from you")

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
    arduino_serial.close()
