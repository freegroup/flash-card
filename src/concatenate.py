import cv2
import numpy as np
import shortuuid

heightImg = 1840
widthImg = 900

card_image_front = cv2.resize(cv2.imread("./images/test.png"), (widthImg, heightImg))
card_image_back = cv2.resize(cv2.imread("./images/test.png"), (widthImg, heightImg))
imgBar = np.zeros((heightImg, 15, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED

guid = shortuuid.uuid()
card_image = np.concatenate((card_image_front, imgBar, card_image_back), axis=1)
cv2.imwrite('card_{}.png'.format(guid), card_image)
