#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dlib
import cv2
import face_recognition
import sys


# In[ ]:


# Start video capture for webcam -  Specifying 0 as an argument fires up the webcam feed
video_capture = cv2.VideoCapture(0)


# In[ ]:


# You can get the mmod_human_face_detector.dat file from http://dlib.net/files/mmod_human_face_detector.dat.bz2"


# In[ ]:


cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')


# In[ ]:


while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    dets = cnn_face_detector(rgb_small_frame, 1)
    for det in dets:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top = 4 * det.rect.top()
        right = 4 * det.rect.right()
        bottom = 4 * det.rect.bottom()
        left = 4 * det.rect.left()
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
    cv2.imshow('Dlib_CNN', frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()

