#pragma once
#include "opencv2/opencv.hpp"
#include <vector>

class PointsSelector
{
public:
	PointsSelector(void);
	~PointsSelector(void);

	static std::vector<cv::Point> SelectPointsOnImage(cv::Mat img, const char* windowName = "Select points");
};

