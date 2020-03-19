import os
import cv2
from PIL import Image
import numpy as np
import pickle


base_dir =os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(base_dir,'fa')
face_cascade=cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()

current_id=0
label_id={}
x_trains=[]
y_labels=[]

for root,dirs,files in os.walk(img_dir):
    for file in files:
        if file.endswith('jpg') or file.endswith('png'):
            path = os.path.join(root,file)
            label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            # print(label,path)
            if label in label_id:
                pass
            else:
                label_id[label]=current_id
                current_id+=1
            id_=label_id[label]
            # print(label_id)

            pil_image= Image.open(path).convert("L") #greyscal
            size=(550,550)
            final_image=pil_image.resize(size,Image.ANTIALIAS)
            image_array= np.array(final_image,"uint8")
            # print(image_array)
            faces=face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)
            for (x,y,w,h) in faces:
                roi=image_array[y:y+w,x:x+h]
                x_trains.append(roi)
                y_labels.append(id_)

with open("label.pickle",'wb') as f:
    pickle.dump(label_id,f)

recognizer.train(x_trains,np.array(y_labels))
recognizer.save("trainer.yml")