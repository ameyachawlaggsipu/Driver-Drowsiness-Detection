import cv2
# Used for Image Processing and Capturing
import numpy as np
# Used for converting calulations
import dlib
# Used for detecting face
from imutils import face_utils
# Used for detecting landmarks

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Using default camera the webcam

detector = dlib.get_frontal_face_detector() # Initializing the detector

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # Pretrained dataset to find landmarks

# Variables to record status of the driver
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB) #Distance between 2 points

def blinked(a,b,c,d,e,f): # Based on ratio of opened eye height and width landmark points
    up = compute(b,d) + compute(c,e)
    down = compute(a,f)
    ratio = up/(2.0*down)
    
    if ratio > 0.25:
        return 2
    elif ratio > 0.21 and ratio <= 0.25:
        return 1
    else:
        return 0
    

while True:
    _, frame = cap.read() # Reading Frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Converting to 2d gray scale
    
    faces = detector(gray) # Detecting Faces
    
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        
        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        landmarks = predictor(gray, face) #Detecting Landmarks
        
        landmarks = face_utils.shape_to_np(landmarks) # Reshaping to np array
        
        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[35])
        
        if left_blink == 0 and right_blink == 0 :
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "Sleeping"
        elif left_blink ==1 or right_blink ==1:
            sleep =0 
            active =0
            drowsy+=1
            if drowsy > 6:
                status = "Drowsy"
        else:
            sleep =0
            drowsy =0
            active+=1
            if active > 6:
                status = "Active"

        
        cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,255), 3)   
        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x,y), 1, (255, 0, 0), -1)
        
        cv2.imshow("frame", frame) # Display of Image with landmarks
        cv2.imshow("Result Detector", face_frame) # Display of Image with text  
        
        key = cv2.waitKey(1)
        
        if key == 27:
            break
            
        
