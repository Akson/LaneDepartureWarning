import cv2 as cv
from LaneMarkersModel import LaneMarkersModel
from LaneMarkersModel import normalize
import numpy as np

from Sensor import LaneSensor

stream = cv.VideoCapture("T:\_DIMA_DATA\Video\LaneDepartureWarningTestVideo\converted\out8.avi") #6 7 8
if stream.isOpened() == False:
    print "Cannot open input video"
    exit()

cropArea = [0, 124, 637, 298]

flag, imgFull = stream.read()
img = imgFull[cropArea[1]:cropArea[3], cropArea[0]:cropArea[2]]
yellowLaneModel = LaneMarkersModel()
#yellowLaneModel.InitializeFromImage(cv.GaussianBlur(np.float32(img)/255.0, (5, 5), 2), "Select yellow lane points")
yellowLaneModel.InitializeFromImage(np.float32(img)/255.0, "Select yellow lane points")

line1Start = np.array([2, 148])
line1End = np.array([281, 0])

line1Start = np.array([71, 163])
line1End= np.array([303, 3])

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
    cv.imshow("input", imgFull)
    img = np.float32(imgFull[cropArea[1]:cropArea[3], cropArea[0]:cropArea[2]])/255.0
    #imgSmoothed = cv.GaussianBlur(np.float32(imgFull[cropArea[1]:cropArea[3], cropArea[0]:cropArea[2]])/255.0, (5, 5), 2)

    #convert to HSV
    hsv = np.float32(cv.cvtColor(img, cv.COLOR_RGB2HSV))
    '''
    cv.imshow("H", normalize(hsv[:,:,0]))
    cv.imshow("S", normalize(hsv[:,:,1]))
    cv.imshow("V", normalize(hsv[:,:,2]))
    
    
    #RGB distance
    distRgb = abs(img-yellowLaneModel.avgRGB)
    distGray = np.sqrt(distRgb[:, :, 0]*distRgb[:, :, 0] + distRgb[:, :, 1]*distRgb[:, :, 1] + distRgb[:, :, 2]*distRgb[:, :, 2]) 
    cv.imshow("distRGB", normalize(distGray))

    #HSV distance
    distHSV = abs(hsv-yellowLaneModel.avgHSV)
    cv.imshow("dH", normalize(distHSV[:, :, 0]))
    cv.imshow("dS", normalize(distHSV[:, :, 1]))
    cv.imshow("dV", normalize(distHSV[:, :, 2]))
    '''
    canny = cv.Canny(cv.cvtColor(np.uint8(img*255), cv.COLOR_RGB2GRAY), 70, 170)
    '''
    cv.imshow("canny", canny)

    cv.imshow("canny dH", cv.Canny(np.uint8(normalize(distHSV[:, :, 0])*255), 200, 250))
    cv.imshow("canny dS", cv.Canny(np.uint8(normalize(distHSV[:, :, 1])*255), 200, 250))
    cv.imshow("canny dV", cv.Canny(np.uint8(normalize(distHSV[:, :, 2])*255), 200, 250))
    
    cv.imshow("ThreshRGB", cv.threshold(np.uint8(distGray*255.0), 30, 255, cv.THRESH_BINARY_INV)[1])
    threshH = cv.threshold(np.uint8((distHSV[:, :, 0])), 10, 255, cv.THRESH_BINARY_INV)[1]
    cv.imshow("ThreshH", threshH)
    cv.imshow("ThreshS", cv.threshold(np.uint8((distHSV[:, :, 1]*255)), 30, 255, cv.THRESH_BINARY_INV)[1])
    cv.imshow("ThreshV", cv.threshold(np.uint8((distHSV[:, :, 2]*255)), 30, 255, cv.THRESH_BINARY_INV)[1])

#    HSVSum = normalize(distHSV[:, :, 0]) + normalize(distHSV[:, :, 1]) + normalize(distHSV[:, :, 2])
    Sum = (1.0 - normalize(1.0*normalize(distGray) + 1.0*normalize(distHSV[:, :, 0]) + 0.0*normalize(distHSV[:, :, 1]) + 0.0*normalize(distHSV[:, :, 2])))
    cv.imshow("Sum", normalize(Sum)*255) 

    Probability = yellowLaneModel.lineProbabilityMap * (normalize(Sum))
    cv.imshow("Probability", normalize(Probability)) 
    
    #try to find yellow line
#    cv.imshow("filteredH", cv.medianBlur(threshH, 7))
    threshP = cv.threshold(np.uint8(Probability*255), 200, 1, cv.THRESH_BINARY)[1]
    cv.imshow("ThreshP", threshP*255)
    
    edges = cv.medianBlur(cv.dilate(threshP, None),5)
    cv.imshow("edges", edges*canny)
    
    yellowLaneModel.UpdateModelFromMask(threshP, img, hsv)

    moments = cv.moments(threshP)
    if moments['m00']>1e-5:
        xm = int(moments['m10']/moments['m00'])
        ym = int(moments['m01']/moments['m00'])
    
    xline = canny[ym,:]
    xr=xm
    while (xline[xr]==0 and xr<len(xline)): xr+=1
    xl=xm
    while (xline[xl]==0 and xl>0): xl-=1
    #print xm, ym, xl, xr
    
    
    flooded = cv.dilate(np.uint8(canny), None)
    largeMask = np.zeros((img.shape[0]+2, img.shape[1]+2), np.uint8)
    largeMask[:] = 0
    flags = 8&cv.FLOODFILL_FIXED_RANGE
    cv.floodFill(flooded, largeMask, (xl, ym), (0, 255, 0), 10, 10, flags)
    mask = largeMask[1:largeMask.shape[0]-1, 1:largeMask.shape[1]-1]
    cv.imshow("Flooded Canny", mask*canny)

    cv.circle(outputImg, (xl, ym), 2, [100, 0, 0])
    cv.circle(outputImg, (xr, ym), 2, [100, 0, 0])
    
    rgbLaneError = np.abs(img - [0.9382180306192947, 0.989098653809665, 0.9846667443236259])
    rgbLaneDistance = np.sqrt(np.square(rgbLaneError[:,:,0])+np.square(rgbLaneError[:,:,1])+np.square(rgbLaneError[:,:,2]))
    '''
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
    
    cv.imshow("Output", outputImg)
    cv.imshow("Output full", outputFull)
    
cv.destroyAllWindows()