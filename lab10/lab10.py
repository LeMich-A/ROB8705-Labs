# Source https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
# need video reader too 
#Author Michel Akpro , Neelanjan Mukherji

#This code intends to mimic optical flow within a video.
#Optical flow is the apparent motion pattern of picture objects between two successive frames 
#that results from camera or object movement. It's a 2D vector field where each vector represents 
#a displacement vector that illustrates how points move from the first to the second frame.

import numpy as np
import glob

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


import numpy as np
import cv2 as cv

# Replace 'filename' with the path to your video file, file to video taken
filename = "lexusis300.mov"
cap = cv.VideoCapture(filename)

# params for ShiTomasi corner detection: https://docs.opencv.org/3.4/d4/d8c/tutorial_py_shi_tomasi.html
feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Create some random colors for visualization
color = np.random.randint(0, 255, (100, 3))

# Take the first frame and find corner points in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# Loop through each frame in the video
while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    
    # Convert the frame to grayscale
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv.add(frame, mask)
        cv.imshow('frame', img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    # Update the previous frame and previous points for the next iteration
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

# Release the video capture object and close all windows
cap.release()
cv.destroyAllWindows()
