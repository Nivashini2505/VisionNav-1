import customtkinter as ctk
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
import threading
import cv2
import numpy as np
import pyttsx3

# Initialize the CustomTkinter app
app = ctk.CTk()
app.geometry("900x600")
app.title("VisionNav Results")

predictions_textbox = ctk.CTkLabel(master=app, text="VisionNav", font=("Arial", 16))
predictions_textbox.place(x=100, y=80)

# Predictions display on the right side
predictions_textbox = ctk.CTkTextbox(master=app, width=300, height=300)
predictions_textbox.place(x=100, y=130)

# Start detection button on the left
start_button = ctk.CTkButton(master=app, text="Start Object Detection", width=200)
start_button.place(x=50, y=60)

# Initialize pyttsx3 for text-to-speech
engine = pyttsx3.init()

# Custom function to handle predictions and print detected objects in a slangy way
def custom_on_prediction(predictions, frame):
    try:
        predictions_textbox.delete("1.0", ctk.END)  # Clear previous predictions
        for prediction in predictions['predictions']:
            object_class = prediction['class']
            confidence = prediction['confidence']
            
            # Create casual, slang-like messages
            if object_class == "chair":
                message = "Yo, there's a chair right in front of ya. Don't trip, man!"
            elif object_class == "table":
                message = "Heads up! A table is chillin' nearby."
            elif object_class == "dog":
                message = "Hey, there's a cute doggo in front of you. Keep cool!"
            elif object_class == "person":
                message = "Uh oh, there's a person right ahead, bro."
            else:
                message = f"Whoa, I see a {object_class}. Confidence level: {confidence:.2f}. Stay sharp!"

            # Insert predictions into the textbox
            predictions_textbox.insert(ctk.END, message + "\n")
            predictions_textbox.see(ctk.END)  # Scroll to the end of the textbox
            print(message)

            # Make the bot speak the slang message
            engine.say(message)
            engine.runAndWait()

        # Render the bounding boxes on the frame
        render_boxes(predictions, frame)

        # Check if the frame is valid and display it (if needed)
        if frame is not None and isinstance(frame, np.ndarray):
            cv2.imshow('Predictions', frame)

        # Press 'q' to exit the detection
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()

    except Exception as e:
        print(f"Error in custom_on_prediction: {e}")

# Function to start detection in a new thread
def start_detection():
    def detection_thread():
        # Initialize a pipeline object
        try:
            pipeline = InferencePipeline.init(
                model_id="living-room-items/1",  # Replace with your Roboflow model ID
                video_reference=0,  # 0 to use the built-in webcam
                api_key="G0HdUVNRRR2g5eGLP20Z",  # Your Roboflow API key
                on_prediction=custom_on_prediction,  # Function to handle predictions
            )
            pipeline.start()
            pipeline.join()

        except Exception as e:
            print(f"Error initializing pipeline: {e}")
        finally:
            cv2.destroyAllWindows()

    # Start the detection thread
    threading.Thread(target=detection_thread, daemon=True).start()

# Link the button to the start detection function
start_button.configure(command=start_detection)

# Start the Tkinter main loop
app.mainloop()
