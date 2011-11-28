#pragma once
#include "opencv2/opencv.hpp"

class LineMarkerModel
{
public:
	LineMarkerModel(void);
	~LineMarkerModel(void);

public:
	cv::Scalar m_baseColor;
};

