import cv2
import json
import os
import numpy as np

# Load the DNN model for face detection
model_file = "res10_300x300_ssd_iter_140000.caffemodel"
config_file = "deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(config_file, model_file)

# Function to load names from the JSON file
def load_name_dict(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}

video = cv2.VideoCapture(0)

# Load the face recognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

# Load name dictionary
name_dict = load_name_dict('name_dict.json')

# Function to detect faces using DNN
def detect_faces_dnn(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            x = max(0, x)
            y = max(0, y)
            x1 = min(w, x1)
            y1 = min(h, y1)
            if x1 > x and y1 > y:
                faces.append((x, y, x1 - x, y1 - y))
    return faces


# Main loop for real-time video processing
while True:
    ret, frame = video.read()
    if not ret:
        break
    
    faces = detect_faces_dnn(frame)

    for (x, y, w, h) in faces:
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if w > 0 and h > 0 and gray[y:y+h, x:x+w].size > 0:
            try:
                serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
                if conf < 50:
                    user_id = str(serial)
                    name = name_dict.get(user_id, "Unknown")
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
                    cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
                    cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)
            except Exception as e:
                print(f"Error in recognition: {e}")

    # Display the frame
    cv2.imshow("Frame", frame)
    cv2.setWindowProperty("Frame", cv2.WND_PROP_TOPMOST, 1)

    # Exit the loop if 'q' is pressed
    k=cv2.waitKey(1)
    if k==ord("q"):
        break

video.release()
cv2.destroyAllWindows()
