'''
Created on Nov 27, 2011

@author: Dima
'''

import cv2 as cv
import numpy as np

class LaneSensor():
    def __init__(self):
        '''
        Constructor
        '''
    xPos = 0
    yPos = 0
    width = 0
    def SetGeometry(self, position, width):
        self.xPos = position[0]-width/2
        self.yPos = position[1] 
        self.width = width
    def DrawGeometry(self, img):
        cv.line(img, (self.xPos, self.yPos), (self.xPos+self.width, self.yPos), [0, 0, 255])
    
    lineRGB = [0, 0, 0]
    lineHSV = [0, 0, 0]
    roadRGB = [0, 0, 0]
    roadHSV = [0, 0, 0]
    def InitializeModel(self, linergb, linehsv, roadrgb, roadhsv):
        self.lineRGB = linergb
        self.lineHSV = linehsv
        self.roadRGB = roadrgb
        self.roadHSV = roadhsv
        
    def CalculatePixelsProperties(self, rgbGlobal, hsvGlobal):
        rgb = rgbGlobal[self.yPos, self.xPos:(self.xPos+self.width), :]
        hsv = hsvGlobal[self.yPos, self.xPos:(self.xPos+self.width), :]
        reliability = np.ones_like(rgb)
        probability = np.ones_like(rgb)
        return (probability, reliability)
        
    def UpdatePositionBasedOnCanny(self, cannyGlobal):
        canny = cannyGlobal[self.yPos, :]
        xr=self.xPos+self.width/2
        while (xr<canny.shape[0] and canny[xr]==0): xr+=1
        xl=self.xPos+self.width/2
        while (xl>0 and canny[xl]==0): xl-=1
        self.xPos = (xl+xr)/2-self.width/2
