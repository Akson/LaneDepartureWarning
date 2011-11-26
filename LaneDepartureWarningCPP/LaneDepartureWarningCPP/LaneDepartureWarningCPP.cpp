#include "stdafx.h"
#include "opencv2/opencv.hpp"

using namespace cv;

int main(int, char**)
{
	VideoCapture cap("T:\\_DIMA_DATA\\Video\\LaneDepartureWarningTestVideo\\2.wmv"); // open the default camera
	if(!cap.isOpened()) return -1;	//cannot open a stream

	namedWindow("edges",1);
	for(;;)
	{
		Mat frame;
		if(cap.read(frame)==false) break;	//end of the video
		imshow("edges", frame);
		if(waitKey(1) == 27) break;	//user pressed ESC
	}
	return 0;
}