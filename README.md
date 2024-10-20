
# Face Recognition System

A Python-based face recognition system using OpenCV and LBPH (Local Binary Pattern Histogram) for training and recognizing faces from images. The system leverages deep learning-based face detection and an LBPH algorithm for recognizing individuals from a dataset.

## Features

- **Face Detection**: Utilizes a pre-trained DNN model for detecting faces in real-time.
- **Face Recognition**: Recognizes faces using the LBPH face recognizer.
- **Dataset Collection**: Automates the collection of face data from the camera.
- **Training**: Trains the face recognizer on a dataset of images.
- **Real-time Recognition**: Recognizes individuals from the live camera feed.

## Requirements

To run this project, ensure you have the following dependencies installed:

```bash
colorama==0.4.6
numpy==2.1.1
opencv-contrib-python==4.10.0.84
opencv-python==4.10.0.84
pillow==10.4.0
PyYAML==6.0.2
tqdm==4.66.5
```

You can install them using `pip`:

```bash
pip install -r requirements.txt
```

## Dataset Collection

1. Run the script to collect face data from the webcam:

```bash
python collect_faces.py
```

- The system will prompt you to enter your name, and it will start capturing images of your face.
- Press `q` to stop data collection.

Collected images are saved in the `datasets/` directory.

## Training the Model

After collecting face data, train the model using the following command:

```bash
python train_model.py
```

- The face recognizer will process the images and store the trained model in `Trainer.yml`.

## Real-Time Face Recognition

To run the real-time face recognition system:

```bash
python recognize_faces.py
```

- The system will use your webcam to detect and recognize faces.
- Recognized individuals' names will be displayed on the video feed.

## Project Structure

```
.
├── collect_faces.py      # Script to collect face data from the webcam
├── train_model.py        # Script to train the LBPH face recognizer
├── recognize_faces.py    # Script for real-time face recognition
├── datasets/             # Directory where face datasets are saved
├── Trainer.yml           # Trained face recognition model
├── name_dict.json        # Stores user IDs and corresponding names
├── requirements.txt      # Python package dependencies
└── README.md             # Project documentation
```

## Model Details

- **Face Detection**: Uses a pre-trained SSD model with a ResNet backbone (`res10_300x300_ssd_iter_140000.caffemodel` and `deploy.prototxt.txt`).
- **Face Recognition**: LBPHFaceRecognizer, which trains on grayscale images and uses local binary patterns to classify faces.

