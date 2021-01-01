import cv2,os
import numpy as np
import face_recognition
from datetime import datetime

path = 'Images'
images = []
classNames = []
mylist = os.listdir(path)
print(mylist)

for cls in mylist:
    current_img = cv2.imread(f'{path}/{cls}')
    images.append(current_img)
    classNames.append(os.path.splitext(cls)[0]) # remove Extendtion

def find_Encoding(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

def markAttendence(name):
    with open("Attendance.csv",'r+') as f:
        mydatalist = f.readlines()
        nameList = []
        for line in mydatalist:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtstring = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dtstring}')

# markAttendence('Elon')

encodeListknown = find_Encoding(images)
print("Encodeing Complete ")


cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25) # this is convert to 1\4 image
    imgs = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facelocCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,facelocCurFrame)

    # checking all images With Local Adrress
    for encodeFace,facelocation in zip(encodeCurFrame,facelocCurFrame):
        matches = face_recognition.compare_faces(encodeListknown,encodeFace)
        facedis = face_recognition.face_distance(encodeListknown,encodeFace)
        matchindex = np.argmin(facedis)

        if matches[matchindex]:
            name = classNames[matchindex].upper()
            print(name) # Name of Person Name If mAtch
            y1,x2,y2,x1 = facelocation
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
            markAttendence(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
