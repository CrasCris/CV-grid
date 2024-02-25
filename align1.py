#Align image without a reference image
import cv2 as cv
import numpy as np

image = cv.imread('Img_sample',cv.IMREAD_GRAYSCALE)
gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 50, 150, apertureSize=3)
contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv.contourArea)
epsilon = 0.1 * cv.arcLength(largest_contour, True)
approx = cv.approxPolyDP(largest_contour, epsilon, True)
canvas = np.zeros_like(image)
cv.drawContours(canvas, [approx], -1, (255, 255, 255), -1)
transform_matrix = cv.getPerspectiveTransform(approx.astype(np.float32), destination_points)
aligned_image = cv.warpPerspective(image, transform_matrix, (image.shape[1], image.shape[0]))

cv.imshow('Aligned Document', aligned_image)
cv.waitKey(0)
cv.destroyAllWindows()
