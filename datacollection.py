import os
import cv2
import json
import numpy as np

# Load the DNN model for face detection
model_file = "res10_300x300_ssd_iter_140000.caffemodel"
config_file = "deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(config_file, model_file)

# Function to create the next id
def get_next_id(path):
    existing_ids = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            name_dict = json.load(f)
            existing_ids = list(map(int, name_dict.keys()))
    return max(existing_ids, default=0) + 1  

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

# Initialize video capture
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: Could not open video file.")
    exit()

if not os.path.exists('datasets'):
    os.makedirs('datasets')

name = input("Enter Your Name: ")
while(name==""):
    print("Please Enter Valid Name!!")
    name = input("Enter Your Name: ")
user_id = get_next_id('name_dict.json')
count = 1

# Main loop for collecting data
while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Detect faces using DNN
    faces = detect_faces_dnn(frame)
    
    for (x, y, w, h) in faces:
        count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'datasets/User.{str(user_id)}.{str(count)}.jpg', gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    # Display the frame
    cv2.imshow("Frame", frame)
    cv2.setWindowProperty("Frame", cv2.WND_PROP_TOPMOST, 1)

    k = cv2.waitKey(1)
    if count > 500 or k == ord('q'):
        break

# Release the video capture and close windows
video.release()
cv2.destroyAllWindows()

# Store the name with the associated user_id in a JSON file
name_dict = {}
if os.path.exists('name_dict.json'):
    with open('name_dict.json', 'r') as f:
        name_dict = json.load(f)

name_dict[str(user_id)] = name  # Store name with the unique user ID

# Write the updated dictionary back to the JSON file
with open('name_dict.json', 'w') as f:
    json.dump(name_dict, f, indent=4)

print("Dataset Collection Done..................")
