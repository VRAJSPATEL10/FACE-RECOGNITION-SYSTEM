import cv2
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# Load the face recognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()

path="datasets"

def getImageID(path):
    imagePath = [os.path.join(path, f) for f in os.listdir(path)]
    faces=[]
    ids=[]
    for imagePaths in tqdm(imagePath, desc="Processing images", unit="image"):
        faceImage = Image.open(imagePaths).convert('L')
        faceNP = np.array(faceImage)
        Id= (os.path.split(imagePaths)[-1].split(".")[1])
        Id=int(Id)
        faces.append(faceNP)
        ids.append(Id)
    return ids, faces

IDs, facedata = getImageID(path)
recognizer.train(facedata, np.array(IDs))
recognizer.write("Trainer.yml")
cv2.destroyAllWindows()
print("Training Completed............")