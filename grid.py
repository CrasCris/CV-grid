import cv2 as cv
import numpy as np

image = cv.imread('C:/Users/criju/OneDrive/Escritorio/Codigos/CVProject/EjemplosTablas/eje1.jpeg')
mask = np.zeros(image.shape, dtype=np.uint8)
gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

# Detect only grid
cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv.contourArea(c)
    if area > 10000:
        cv.drawContours(mask, [c], -1, (255,255,255), -1)

mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
mask = cv.bitwise_and(mask, thresh)

# Find horizontal lines
horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (55,1))
detect_horizontal = cv.morphologyEx(mask, cv.MORPH_OPEN, horizontal_kernel, iterations=2)
cnts = cv.findContours(detect_horizontal, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv.drawContours(image, [c], -1, (0,0,255), 2)

# Find vertical lines
vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,25))
detect_vertical = cv.morphologyEx(mask, cv.MORPH_OPEN, vertical_kernel, iterations=2)
cnts = cv.findContours(detect_vertical, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv.drawContours(image, [c], -1, (0,0,255), 2)


cv.imshow('image', image)
cv.waitKey()