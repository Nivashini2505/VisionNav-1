# Import the necessary modules
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
import cv2

# Custom function to handle predictions and print detected objects
def custom_on_prediction(predictions, frame):
    try:
        # print(f"Predictions: {predictions}")

        for prediction in predictions['predictions']:
            print(f"Prediction: {prediction['class']} and confidence: {prediction['confidence']}")

        
        # Render the bounding boxes on the frame
        render_boxes(predictions, frame)
        
        # Display the frame with predictions
        # TASK: SOME ERROR IS HERE => FIX IT
        '''
        Error in custom_on_prediction: OpenCV(4.8.0) :-1: error: (-5:Bad argument) in function 'imshow'
        > Overload resolution failed:
        >  - mat is not a numpy array, neither a scalar
        >  - Expected Ptr<cv::cuda::GpuMat> for argument 'mat'
        >  - Expected Ptr<cv::UMat> for argument 'mat'
        '''
        
        cv2.imshow('Predictions', frame)
        
        # Add a delay to allow the frame to be displayed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()
        
    except Exception as e:
        print(f"Error in custom_on_prediction: {e}")

# Initialize a pipeline object
try:
    pipeline = InferencePipeline.init(
        model_id="living-room-items/1",  # Replace with your Roboflow model ID
        video_reference=1,  # 0 to use the built-in webcam
        api_key="G0HdUVNRRR2g5eGLP20Z",  # Your Roboflow API key
        on_prediction=custom_on_prediction,  # Use the custom function for predictions
    )

    pipeline.start()
    pipeline.join()
except Exception as e:
    print(f"Error initializing pipeline: {e}")
finally:
    cv2.destroyAllWindows()  # Ensure all OpenCV windows are destroyed