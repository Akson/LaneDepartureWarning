#include "stdafx.h"
#include "opencv2/opencv.hpp"
#include <vector>
#include "PointsSelector.h"

using namespace cv;
using namespace std;

VideoCapture getVideoCapture(int testCase = 0, float posSec = 0)
{
	VideoCapture cap;

	if(testCase > 0) 
	{
		char fileName[1024];
		sprintf(fileName, "T:\\_DIMA_DATA\\Video\\LaneDepartureWarningTestVideo\\%d.wmv", testCase);
		cap.open(fileName); // open the video file
		if(!cap.isOpened()) throw "Cannot open file";
		cap.set(CV_CAP_PROP_POS_MSEC, posSec*1000);
	}
	else 
	{
		cap.open(-testCase); // open the camera stream
	}

	return cap;
}

int main(int, char**)
{
	VideoCapture cap;
	try {cap = getVideoCapture(8, 0);}
	catch (char* e) {printf("ERROR: %s", e); return -1;}
	
	namedWindow("Input", 1);
	namedWindow("HSV", 1);
	namedWindow("HSV1", 1);
	namedWindow("Gray", 1);
	namedWindow("ColorDiff", 1);
	namedWindow("HDiff", 1);
	namedWindow("SDiff", 1);
	
	Mat inputFrame;

	//Read first frame and select some initial data
	cap.read(inputFrame);
	inputFrame.locateROI(Size(100,100), Point(50,100));

	//calculate the average color of a yellow line
	auto yellowLanePoints = PointsSelector::SelectPointsOnImage(inputFrame, "Select yellow lane points");
	Scalar avgColor;
	for (auto point = yellowLanePoints.begin(); point != yellowLanePoints.end(); point++)
	{
		int x = (*point).x;
		int y = (*point).y;
		auto color = inputFrame.at<Vec3b>(y, x);
		for(int i=0; i<3; i++) avgColor[i]+=color[i];
	}
	for(int i=0; i<3; i++) avgColor[i]/=yellowLanePoints.size();
	Mat yellowMarkerColor(1, 1, CV_8UC3, avgColor);
	Mat yellowMarkerColorHSV;
	cvtColor(yellowMarkerColor, yellowMarkerColorHSV, CV_RGB2HSV);
	Scalar avgHSV(yellowMarkerColorHSV.at<Vec3b>(0,0)[0], yellowMarkerColorHSV.at<Vec3b>(0,0)[1], yellowMarkerColorHSV.at<Vec3b>(0,0)[2]);

	Mat frameGray;
	Mat frameHSV;
	vector<Mat> frameChannels;
	Mat colorDiff;
	Mat HSVDiff;
	vector<Mat> frameHSVDiffChannels;

	do
	{
		if(cap.read(inputFrame)==false) break;	//end of the video
		inputFrame.locateROI(Size(100,100), Point(50,100));
		inputFrame.locateROI()

		cvtColor(inputFrame, frameHSV, CV_RGB2HSV);
		cvtColor(inputFrame, frameGray, CV_RGB2GRAY);
		split(frameHSV, frameChannels);
		imshow("Input", inputFrame);
		imshow("Gray", frameGray);
		imshow("HSV", frameChannels[0]);
		imshow("HSV1", frameChannels[1]);

		absdiff(inputFrame, avgColor, colorDiff);
		imshow("ColorDiff", colorDiff);

		absdiff(frameHSV, avgHSV, HSVDiff);
		split(HSVDiff, frameHSVDiffChannels);
		imshow("HDiff", frameHSVDiffChannels[0]);
		imshow("SDiff", frameHSVDiffChannels[1]);
	}
	while(waitKey(1) != 27);	//user pressed ESC

	return 0;
}