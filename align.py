# Libs
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib as plt

# #Model image
model_img = cv.imread('path_model_img',cv.IMREAD_GRAYSCALE)

# #Sample image
fit_img = cv.imread('path_sample_img',cv.IMREAD_GRAYSCALE)

#Review if the code need to be in color frits and then convert to grayscale

# #Using ORB features and compute descriptors
Max_num_features= 500
orb = cv.ORB_create(max_num_features)
keypoints1,descriptors1 = orb.detectAndCompute(model_img,None)
keypoints2,descriptors2 = orb.detectAndCompute(fit_img,None)

# #Display images
model_display = cv.drawKeypoints(model_img,keypoints1,outImage=np.array([]),color=(255,0,0),flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
sample_display = cv.drawKeypoints(fit_img,keypoints2,outImage=np.array([]),color=(255,0,0),flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
plt.figure(figsize=[20,10])
plt.subplot(121); plt.axis('off');plt.imshow(model_display);plt.title('Modelo')
plt.subplot(122); plt.axis('off');plt.imshow(sample_display);plt.title('Sample')

#Match keypoints in the two image

#Features
matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = matcher.match(descriptors1,descriptors2,None)

#Sort matches by score
matches.sort(key=lambda x: x.distance,reverse=False)

#Remove bad matches
numGoodMatches = int(len(matches)*0.1)
matches = matches[:numGoodMatches]

#Only draw the top matches
img_matches = cv.drawMatches(model_img,keypoints1,fit_img,keypoints2,matches,None)

plt.figure(figsize=[40,10])
plt.imshow(img_matches); plt.axis('off');plt.title('Sample form')

#Find Homography
points1 = np.zeros((len(matches),2),dtype=np.float32)
points2 = np.zeros((len(matches),2),dtype=np.float32)

for i,match in enumerate(matches):
    points1[i,:] = keypoints1[match.queryIdx].pt
    points2[i,:] = keypoints2[match.queryIdx].pt

#Find Homography
h,mask =cv.findHomography(points2,points1,cv.RANSAC)

height,width,channels = model_img
fit_reg = cv.warpPerspective(fit_img,h,(height,width))

#Display results
plt.figure(figsize=[20,10])
plt.subplot(121); plt.imshow(model_img);plt.axis('off');plt.title('Model form')
plt.subplot(122); plt.imshow(fit_reg);plt.axis('off');plt.title('Sample Final result')
