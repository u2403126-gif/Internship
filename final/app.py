import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime

st.title("Face Attendance System")

menu = ["Register Face", "Train Model", "Mark Attendance", "View Attendance"]
choice = st.sidebar.selectbox("Menu", menu)

if not os.path.exists("faces"):
    os.makedirs("faces")

if not os.path.exists("trainer"):
    os.makedirs("trainer")

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

if choice == "Register Face":

    st.header("Register New Student")

    name = st.text_input("Enter Student Name")

    if st.button("Capture Faces"):

        cam = cv2.VideoCapture(0)

        count = 0

        while True:

            ret, frame = cam.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:

                count += 1

                face = gray[y:y+h, x:x+w]

                cv2.imwrite(
                    f"faces/{name}.{count}.jpg",
                    face
                )

                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

            cv2.imshow("Capturing Faces", frame)

            if cv2.waitKey(1) == 13 or count >= 30:
                break

        cam.release()

        cv2.destroyAllWindows()

        st.success("Face Registered Successfully")


if choice == "Train Model":

    st.header("Training Model")

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    ids = []

    imagePaths = [os.path.join("faces", f) for f in os.listdir("faces")]

    names = {}
    current_id = 0

    for imagePath in imagePaths:

        gray_img = Image.open(imagePath).convert('L')

        img_numpy = np.array(gray_img, 'uint8')

        name = os.path.split(imagePath)[-1].split(".")[0]

        if name not in names:
            names[name] = current_id
            current_id += 1

        id = names[name]

        faces.append(img_numpy)
        ids.append(id)

    recognizer.train(faces, np.array(ids))

    recognizer.save("trainer/trainer.yml")

    np.save("trainer/names.npy", names)

    st.success("Model Trained Successfully")


if choice == "Mark Attendance":

    st.header("Live Attendance")

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.read("trainer/trainer.yml")

    names = np.load("trainer/names.npy", allow_pickle=True).item()

    reverse_names = {v:k for k,v in names.items()}

    cam = cv2.VideoCapture(0)

    marked = []

    while True:

        ret, frame = cam.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:

            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 70:

                name = reverse_names[id]

                if name not in marked:

                    now = datetime.now()

                    date = now.strftime("%Y-%m-%d")

                    time = now.strftime("%H:%M:%S")

                    df = pd.read_csv("attendance.csv")

                    new_row = pd.DataFrame({
                        "Name":[name],
                        "Date":[date],
                        "Time":[time]
                    })

                    df = pd.concat([df, new_row], ignore_index=True)

                    df.to_csv("attendance.csv", index=False)

                    marked.append(name)

                cv2.putText(
                    frame,
                    name,
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )

            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        cv2.imshow("Attendance", frame)

        if cv2.waitKey(1) == 13:
            break

    cam.release()

    cv2.destroyAllWindows()

    st.success("Attendance Marked")


if choice == "View Attendance":

    st.header("Attendance Report")

    df = pd.read_csv("attendance.csv")

    st.dataframe(df)