import numpy as np 
import cv2
import os 
from matplotlib import pyplot as plt 

# Initialize Max Match
maxMatches = 0

# Initialize directory
directoryScn = "scene"

# Read the image that we want to search
img_obj = cv2.imread("dog.png")

# Gaussian Blur Filter
img_obj = cv2.GaussianBlur(img_obj, (5, 5), 5)

# Convert each images in obj to GRAY
# because histogram equalize can be applied on GRAYSCALE image
img_obj = cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)

# Histogram Equalize
img_obj = cv2.equalizeHist(img_obj)

for filename in os.listdir(directoryScn):
    # Read the image in scene folder
    img_scene = cv2.imread(directoryScn+"/"+filename)

    # SURF
    surf = cv2.xfeatures2d.SURF_create()

    kp_obj, des_obj = surf.detectAndCompute(img_obj, None)
    kp_scene, des_scene = surf.detectAndCompute(img_scene, None)

    # Initialize parameters for FLANN
    index_params = dict(algorithm=0)
    search_params = dict(checks=50)

    # FLANN
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_obj, des_scene, k=2)

    matchesMask = [[0,0] for i in range(len(matches))]

    matchesCount = 0
    for idx, (m,n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[idx] = [1,0]
            matchesCount+=1

    if matchesCount > maxMatches:
        maxMatches = matchesCount
        result = cv2.drawMatchesKnn(img_obj, kp_obj, img_scene, kp_scene, matches, None, matchColor = [0,255,0], singlePointColor = [255,0,0], matchesMask = matchesMask)

# Show Result images
plt.imshow(result)

plt.show()

cv2.waitKey(0)