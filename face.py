import numpy as np
import cv2
import pickle

labels={}
with open("label.pickle",'rb') as f:
    g_labels=pickle.load(f)
    labels={v:k for k,v in g_labels.items()}

face_cascade=cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml')
cap=cv2.cv2.VideoCapture(0)
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
while True:
    ret, frame = cap.read()
    grey= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(grey,scaleFactor=1.5,minNeighbors=5)
    for (x,y,h,w) in faces:
        # print(x,y,w,h)
        roi_grey=grey[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        id_,conf = recognizer.predict(roi_grey)
        if conf >=45 :
            print(id_)
            print(labels[id_])
            font=cv2.FONT_HERSHEY_COMPLEX
            name= labels[id_]
            color=(255,255,255)
            stroke=2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
        # cv2.imwrite('img_name',roi_color)  #this will save image of origin

        #creating a rectangle on face
        color=(255,0,0) 
        stroke=2
        end_cordi_x=x+w
        start_cordi_y=y+h
        cv2.rectangle(frame,(x,y),(end_cordi_x,start_cordi_y),color,stroke)

    cv2.cv2.imshow('frame',frame)
    if cv2.cv2.waitKey(20) & 0xFF ==ord('q'):
        break
    