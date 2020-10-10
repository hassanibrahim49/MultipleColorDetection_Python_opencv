# Python code for Multiple Color Detection
import numpy as np
import cv2

Flags = {'Red_Flag': 0, 'Green_Flag': 0, 'Blue_Flag': 0, 'Yellow_Flag': 0, 'Orange_Flag': 0}

hsv_color_boundaries = [
    ([0, 100, 100], [7, 255, 255]),  # Red
    ([10, 100, 100], [17, 255, 255]),  # orange
    ([23, 80, 80], [40, 255, 255]),  # Yellow
    ([55, 160, 40], [90, 255, 255]),  # Green
    ([107, 130, 100], [130, 255, 255])  # Blue
]

# Capturing video through webcam
webcam = cv2.VideoCapture(0)
webcam.set(3, 1280)
webcam.set(4, 720)

while True:

    # Reading the video from the webcam in image frames
    _, imageFrame = webcam.read()

    # Convert the imageFrame in BGR(RGB color space) to
    # HSV(hue-saturation-value) color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    """*********************************************************************************"""
    """ Color Ranges """
    # (Red) Color Range
    red_lower = np.array([0, 100, 100], np.uint8)
    red_upper = np.array([7, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    # (Orange & Brown) Color Range
    orange_lower = np.array([10, 100, 100], np.uint8)
    orange_upper = np.array([17, 255, 255], np.uint8)
    orange_mask = cv2.inRange(hsvFrame, orange_lower, orange_upper)

    # (Yellow) Color Range
    yellow_lower = np.array([23, 80, 80], np.uint8)
    yellow_upper = np.array([40, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

    # (Green) Color Range
    green_lower = np.array([55, 160, 40], np.uint8)
    green_upper = np.array([90, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    # (Blue) Color Range
    blue_lower = np.array([107, 130, 100], np.uint8)
    blue_upper = np.array([130, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    """*********************************************************************************"""
    """ Mask """
    # Morphological Transform, Dilation, for each color and bitwise_and operator
    # between imageFrame and mask determines, to detect only that particular color
    kernal = np.ones((5, 5), "uint8")

    # For red color
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(imageFrame, imageFrame, mask=red_mask)

    # For Orange color
    orange_mask = cv2.dilate(orange_mask, kernal)
    res_orange = cv2.bitwise_and(imageFrame, imageFrame, mask=orange_mask)

    # For Yellow color
    yellow_mask = cv2.dilate(yellow_mask, kernal)
    res_yellow = cv2.bitwise_and(imageFrame, imageFrame, mask=yellow_mask)

    # For green color
    green_mask = cv2.dilate(green_mask, kernal)
    res_green = cv2.bitwise_and(imageFrame, imageFrame, mask=green_mask)

    # For blue color
    blue_mask = cv2.dilate(blue_mask, kernal)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame, mask=blue_mask)

    """*********************************************************************************"""
    # Creating contour to track red color
    _, contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 50000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imageFrame, "Red", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
            Flags['Red_Flag'] = 1
            print('red')

    # Creating contour to track orange color
    _, contours, hierarchy = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 50000:
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (207, 104, 0), 2)
            cv2.putText(imageFrame, "Orange", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (207, 104, 0))
            Flags['Orange_Flag'] = 1
            print('Orange')

    # Creating contour to track yellow color
    _, contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 50000:
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (207, 200, 0), 2)
            cv2.putText(imageFrame, "Yellow", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (207, 200, 0))
            Flags['Yellow_Flag'] = 1
            print('Yellow')

    # Creating contour to track green color
    _, contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 50000:
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imageFrame, "Green", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
            Flags['Green_Flag'] = 1
            print('Green')

    # Creating contour to track blue color
    _, contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 10000:
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(imageFrame, "Blue", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))
            Flags['Blue_Flag'] = 1
            print('Blue')
            
    """*********************************************************************************"""
    # Program Termination
    cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        Flags_Found = [i for i, j in Flags.items() if j == 1]
        if not Flags_Found:
            print("No flags found")
        else:
            print(Flags_Found)

        webcam.release()
        cv2.destroyAllWindows()
        break
