import cv2

cap = cv2.VideoCapture("out1.avi")
if cap.isOpened() == False:
    print "Cannot open input video"
    exit()

cv2.namedWindow("input")

while(True):
    f, img = cap.read()
    cv2.imshow("input", img)
    cv2.waitKey(1)

cv2.destroyWindow("input")
