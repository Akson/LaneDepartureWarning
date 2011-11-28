#include "StdAfx.h"
#include "PointsSelector.h"

using namespace std;
using namespace cv;

PointsSelector::PointsSelector(void)
{
}

PointsSelector::~PointsSelector(void)
{
}

void onMouse( int event, int x, int y, int, void* data)
{
	if( event != CV_EVENT_LBUTTONDOWN )
		return;

	vector<Point> *res = (vector<Point>*)data;
	res->push_back(Point(x, y));
	printf("Added point %d, %d\n", x, y);
}

std::vector<cv::Point> PointsSelector::SelectPointsOnImage( cv::Mat img, const char* windowName /*= "Select points"*/ )
{
	namedWindow(windowName);
	vector<Point> results;
	setMouseCallback( windowName, onMouse, &results );
	imshow(windowName, img);
	waitKey(0);
	destroyWindow(windowName);
	return results;
}
