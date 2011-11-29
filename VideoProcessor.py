import cv2 as cv
from LaneMarkersModel import LaneMarkersModel
from LaneMarkersModel import normalize
import numpy as np

from Sensor import LaneSensor

# uncompressed YUV 4:2:0 chroma subsampled
fourcc = cv.cv.CV_FOURCC('M','J','P','G')
videoWriter = cv.VideoWriter('out.avi', fourcc, 30, (640, 480), 1)

stream = cv.VideoCapture("T:\_DIMA_DATA\Video\LaneDepartureWarningTestVideo\converted\out6.avi") #6 7 8
if stream.isOpened() == False:
    print "Cannot open input video"
    exit()

cropArea = [0, 124, 637, 298]

flag, imgFull = stream.read()
img = imgFull[cropArea[1]:cropArea[3], cropArea[0]:cropArea[2]]
yellowLaneModel = LaneMarkersModel()
#yellowLaneModel.InitializeFromImage(cv.GaussianBlur(np.float32(img)/255.0, (5, 5), 2), "Select yellow lane points")
yellowLaneModel.InitializeFromImage(np.float32(img)/255.0, "Select yellow lane points")

line1Start = np.array([35, 128])
line1End = np.array([220, 32])

#line1Start = np.array([71, 163])
#line1End= np.array([303, 3])

leftLineSensors = []
sensorsNumber = 70
sensorsWidth = 50
for iSensor in range(0, sensorsNumber):
    sensor = LaneSensor()
    pos = line1Start + iSensor*(line1End-line1Start)/(sensorsNumber+1)
    sensor.SetGeometry(pos, sensorsWidth)
    sensor.InitializeModel(yellowLaneModel.avgRGB, yellowLaneModel.avgHSV, (0.8318770212301404, 0.784796499543384, 0.6864621111668014), (41.02017792349725, 0.17449159984689502, 0.832041028240797))
#    sensor.InitializeModel((0.9382180306192947, 0.989098653809665, 0.9846667443236259), (172.64653449609088, 0.05346018975006449, 0.9909903692872556), (0.8318770212301404, 0.784796499543384, 0.6864621111668014), (41.02017792349725, 0.17449159984689502, 0.832041028240797))
    leftLineSensors.append(sensor) 

leftLineModel = np.poly1d(np.polyfit([line1Start[1], line1End[1]], [line1Start[0], line1End[0]], 1)) 

testLeftLineY = 129
testLeftLineXOkColor = np.array([0,255,0])/1.0
testLeftLineXAlert = 130
testLeftLineXAlertColor = np.array([0,128,255])/1.0
testLeftLineXDanger = 200
testLeftLineXDangerColor = np.array([0,0,255])/1.0

while(cv.waitKey(1) != 27):
    #read and crop
    flag, imgFull = stream.read()
    if flag == False: break #end of video
    #cv.imshow("input", imgFull)
    img = np.float32(imgFull[cropArea[1]:cropArea[3], cropArea[0]:cropArea[2]])/255.0
    hsv = np.float32(cv.cvtColor(img, cv.COLOR_RGB2HSV))
    canny = cv.Canny(cv.cvtColor(np.uint8(img*255), cv.COLOR_RGB2GRAY), 70, 170)
 
    #sensors
    outputImg = img.copy()
    outputFull = imgFull.copy()

    laneCoordinatesX = []
    laneCoordinatesY = []
    for sensor in leftLineSensors:
        linesNumber, lineSegments, allSegments = sensor.FindSegments(img, hsv, canny, outputImg, leftLineModel(sensor.yPos))
        if linesNumber == 1:
            sensor.UpdatePositionAndModelFromRegion(img, hsv, lineSegments[0])
            laneCoordinatesY.append((lineSegments[0][0]+lineSegments[0][1])/2)
            laneCoordinatesX.append(sensor.yPos)
#        if linesNumber == 0:
#            sensor.UpdatePositionBasedOnCanny(canny)
            
        sensor.UpdatePositionIfItIsFarAway(leftLineModel(sensor.yPos))
    
    if len(laneCoordinatesX)>0:
        leftLineModel = np.poly1d(np.polyfit(laneCoordinatesX, laneCoordinatesY, 1))
        for sensor in leftLineSensors:
            #cv.circle(outputImg, (laneCoordinatesX[i], laneCoordinatesY[i]), 2, [200, 0, 100], 2)
            cv.circle(outputImg, (int(leftLineModel(sensor.yPos)), sensor.yPos), 2, [100, 0, 200], 1)
    
    #test left lane
    testLeftLineIntersection = int(leftLineModel(testLeftLineY))
    
    #make final output
    lanePosition = 'Ok'
    lanePositionColor = [0, 255, 0]
    if testLeftLineIntersection > testLeftLineXAlert: 
        lanePosition = 'Alert'
        lanePositionColor = testLeftLineXAlertColor
    if testLeftLineIntersection > testLeftLineXDanger: 
        lanePosition = 'Danger'
        lanePositionColor = testLeftLineXDangerColor

    #line model
    cv.line(outputFull, (cropArea[0]+int(leftLineModel(0)), cropArea[1]+0), (cropArea[0]+int(leftLineModel(img.shape[0])), cropArea[1]+img.shape[0]), [255, 0, 0], 2)        
    
    #zones
    cv.line(outputFull, (cropArea[0]+0,cropArea[1]+testLeftLineY) , (cropArea[0]+testLeftLineXAlert,cropArea[1]+testLeftLineY), testLeftLineXOkColor, 2)
    cv.line(outputFull, (cropArea[0]+testLeftLineXAlert,cropArea[1]+testLeftLineY) , (cropArea[0]+testLeftLineXDanger,cropArea[1]+testLeftLineY), testLeftLineXAlertColor, 2)
    cv.line(outputFull, (cropArea[0]+testLeftLineXDanger,cropArea[1]+testLeftLineY) , (cropArea[0]+img.shape[1]/2,cropArea[1]+testLeftLineY), testLeftLineXDangerColor, 2)
    cv.line(outputFull, (cropArea[0]+0,cropArea[1]+testLeftLineY) , (cropArea[0]+img.shape[1],cropArea[1]+testLeftLineY), [0.2,0.2,0.2])
    #intersection circle
    cv.circle(outputFull, (cropArea[0]+testLeftLineIntersection, cropArea[1]+testLeftLineY), 2, lanePositionColor, 3)
    #alerts
    if lanePosition == 'Alert' or lanePosition == 'Danger':
        cv.line(outputFull, (img.shape[1]/2,50), (img.shape[1]/2-25,75), lanePositionColor, 15)
        cv.line(outputFull, (img.shape[1]/2,100), (img.shape[1]/2-25,75), lanePositionColor, 15)
    if lanePosition == 'Danger':
        cv.line(outputFull, (img.shape[1]/2-30,50), (img.shape[1]/2-25-30,75), lanePositionColor, 15)
        cv.line(outputFull, (img.shape[1]/2-30,100), (img.shape[1]/2-25-30,75), lanePositionColor, 15)
    
    #show output
    cv.imshow("Output", outputImg)
    cv.imshow("Output full", outputFull)
    
    #write output
    videoWriter.write(outputFull)
    
cv.destroyAllWindows()