import cv2
import numpy as np
import utility
import time
import shortuuid

########################################################################
webCamFeed = True
pathImage = "1.jpg"
url='http://192.168.178.58:8080'
cap = cv2.VideoCapture(url+"/video")

time.sleep(5)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

heightImg = 1840
widthImg = 900
########################################################################

utility.initializeTrackbars()

card_image_front = None
card_image_back = None
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
imgBar = np.zeros((heightImg, 5, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
card_image_front = imgBlank
card_image_back = imgBlank

def thumb(image):
    return cv2.resize(image, (int(widthImg/3), int(heightImg/4)))

while True:

    if webCamFeed:
        success, img = cap.read()
    else:
        img = cv2.imread(pathImage)
    img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    thres = utility.valTrackbars()  # GET TRACK BAR VALUES FOR THRESHOLDS
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])  # APPLY CANNY BLUR
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)  # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION

    ## FIND ALL COUNTOURS
    imgContours = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)  # DRAW ALL DETECTED CONTOURS

    # FIND THE BIGGEST COUNTOUR
    biggest, maxArea = utility.biggestContour(contours)  # FIND THE BIGGEST CONTOUR
    if biggest.size != 0:
        biggest = utility.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)  # DRAW THE BIGGEST CONTOUR
        imgBigContour = utility.drawRectangle(imgBigContour, biggest, 2)
        pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])  # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        # REMOVE 20 PIXELS FORM EACH SIDE
        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))

        # APPLY ADAPTIVE THRESHOLD
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)

        # Image Array for Display
        imageArray = ([[img, imgThreshold, imgContours, imgWarpColored],
                       [card_image_front, card_image_back, imgBlank, imgBlank]])

    else:
        imageArray = ([[img, imgThreshold, imgContours, imgBlank],
                       [card_image_front, card_image_back, imgBlank, imgBlank]])

    # LABELS FOR DISPLAY
    lables = [["Original", "Threshold", "Contours",  "Cropped Image"],
              ["Front Side", "Back Side", "-",  "-"]]

    stackedImage = utility.stackImages(imageArray, 0.3, lables)
    cv2.imshow("Result", stackedImage)

    ch = cv2.waitKey(1)
    # SAVE IMAGE WHEN 'f' key is pressed
    if ch == ord('f'):
        print("front side")
        card_image_front = imgWarpColored.copy()

    # SAVE IMAGE WHEN 'f' key is pressed
    if ch == ord('b'):
        print("back side")
        card_image_back = imgWarpColored.copy()

    # SAVE IMAGE WHEN 'f' key is pressed
    if ch == ord('s'):
        guid = shortuuid.uuid()
        print("saved")
        card_image = np.concatenate((card_image_front, imgBar, card_image_back), axis=1)
        cv2.imwrite('./cards/Torax/card_{}.png'.format(guid), card_image)
        card_image_front = imgBlank
        card_image_back = imgBlank
