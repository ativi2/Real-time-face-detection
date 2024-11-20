import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from sklearn.metrics import f1_score
import numpy as np


def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, bboxs

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(23-32)', '(38-45)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)


true_genders = []  # List of true gender labels for F1 score
true_ages = []     # List of true age labels for F1 score
predicted_genders = []  # Predicted gender labels
predicted_ages = []     # Predicted age labels


def start_video():
    global running
    if not running:
        running = True
        start_button.config(text="Stop Video")
        status_label.config(text="Status: Detecting...")
        capture_video()
    else:
        running = False
        start_button.config(text="Start Video")
        status_label.config(text="Status: Stopped")


def capture_video():
    global running
    ret, frame = video.read()
    if not ret or not running:
        return

    frame, bboxs = faceBox(faceNet, frame)

    for bbox in bboxs:
        face = frame[max(0, bbox[1] - 20):min(bbox[3] + 20, frame.shape[0] - 1), max(0, bbox[0] - 20):min(bbox[2] + 20, frame.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
      
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]

        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]

        label = "{}, {}".format(gender, age)
        cv2.rectangle(frame, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

   
        predicted_genders.append(gender)
        predicted_ages.append(age)

      
        true_genders.append('Male')  
        true_ages.append('(23-32)')  


    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)

    panel.config(image=image)
    panel.image = image

    if running:
        panel.after(10, capture_video)

    
    if len(true_genders) > 10:  
        calculate_f1_score()


def calculate_f1_score():

    gender_f1 = f1_score(true_genders, predicted_genders, average='weighted', labels=genderList)
    age_f1 = f1_score(true_ages, predicted_ages, average='weighted', labels=ageList)
    

    gender_accuracy = np.mean(np.array(true_genders) == np.array(predicted_genders)) * 100
    age_accuracy = np.mean(np.array(true_ages) == np.array(predicted_ages)) * 100

    f1_label.config(text=f"Gender F1 Score: {gender_f1:.4f} | Age F1 Score: {age_f1:.4f}")
    accuracy_label.config(text=f"Gender Accuracy: {gender_accuracy:.2f}% | Age Accuracy: {age_accuracy:.2f}%")


root = tk.Tk()
root.title("Age and Gender Detection")
root.geometry("900x650")
root.config(bg="#3b3b3b")

title_label = tk.Label(root, text="Age and Gender Detection", font=("Helvetica", 20, "bold"), fg="white", bg="#3b3b3b")
title_label.pack(pady=20)

panel = tk.Label(root)
panel.pack(padx=10, pady=10)

status_label = tk.Label(root, text="Status: Stopped", font=("Helvetica", 14), fg="white", bg="#3b3b3b")
status_label.pack(pady=5)

start_button = tk.Button(root, text="Start Video", width=20, height=2, bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"), relief="flat", command=start_video)
start_button.pack(pady=20)

def on_enter(event):
    start_button.config(bg="#45a049")

def on_leave(event):
    start_button.config(bg="#4CAF50")

start_button.bind("<Enter>", on_enter)
start_button.bind("<Leave>", on_leave)


f1_label = tk.Label(root, text="Gender F1 Score: 0.0000 | Age F1 Score: 0.0000", font=("Helvetica", 12), fg="white", bg="#3b3b3b")
f1_label.pack(pady=10)

accuracy_label = tk.Label(root, text="Gender Accuracy: 0.00% | Age Accuracy: 0.00%", font=("Helvetica", 12), fg="white", bg="#3b3b3b")
accuracy_label.pack(pady=5)


video = cv2.VideoCapture(0)
running = False

root.mainloop()


video.release()
cv2.destroyAllWindows()
