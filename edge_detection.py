import os
import cv2 
import numpy as np 
import matplotlib.image as mpimg

path = "/Levels screens/Level tool screen/"
for i,img in enumerate(os.listdir(path)):
  image = cv2.imread(f"{path}{img}",)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (7, 7), 0)
  _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

  contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Filter contours based on area or other criteria
  filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 15000]

  # Approximate the board outline
  epsilon = 0.01 * cv2.arcLength(filtered_contours[0], True)
  approx = cv2.approxPolyDP(filtered_contours[0], epsilon, True)

  # Get the bounding box of the contour
  x, y, w, h = cv2.boundingRect(approx)

  # Extract the region of interest (ROI) from the original image
  board_roi = image[y:y+h, x:x+w]
  board_roi = board_roi[2:-2,2:-2]
  mpimg.imsave(f'/updated_levels_2/{i}.png', board_roi) 
  #plt.imshow(board_roi)
  #plt.show() 
