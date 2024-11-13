pip install face-recognition

pip install opencv-python-headless

pip install opencv-python 

import cv2
import matplotlib.pyplot as plt

# Initialize the face classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start video capture
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

last_frame = None  # Variable to store the last frame

try:
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        # Draw square boxes around faces
        for (x, y, w, h) in faces:
            # Determine the size of the square (largest dimension of the rectangle)
            side_length = max(w, h)
            
            # Adjust x and y for a centered square box
            x_centered = x + (w - side_length) // 2
            y_centered = y + (h - side_length) // 2

            # Draw a square around the detected face
            cv2.rectangle(frame, (x_centered, y_centered), (x_centered + side_length, y_centered + side_length), (255, 0, 0), 2)

        # Store the frame to display it later
        last_frame = frame

        # Display the resulting frame with matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        plt.title("Face Recognition")
        plt.axis('off')  # Hide axis
        plt.draw()
        plt.pause(0.01)  # Small pause to update the plot

except KeyboardInterrupt:
    # Handle keyboard interrupt gracefully
    print("Face recognition process interrupted.")

finally:
    # Ensure the webcam is released in all cases
    cap.release()
    print("Webcam has been released.")

    # Display the last frame with detected faces using matplotlib
    if last_frame is not None:
        last_frame_rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
        plt.imshow(last_frame_rgb)
        plt.title("Detected Faces")
        plt.axis('off')  # Hide axis
        plt.show()
    else:
        print("No frame was captured.")
