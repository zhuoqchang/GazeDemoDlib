#pragma once

#ifndef GAZEDISPLAY_H
#define GAZEDISPLAY_H

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>		
#include <opencv2/imgproc/imgproc.hpp>

class gazeDisplay {
public:
	gazeDisplay(int w, int h, int bh, int bv);
	~gazeDisplay();

	cv::Mat draw(const cv::Point& p1, const cv::Point& p2);
	void changeType();

private:
	int width, height;
	int hblocks, vblocks;
	std::vector<cv::Scalar> brightColors, darkColors;
	cv::Mat bg0, bg1;
	bool type;
	std::vector<cv::Point2d> regionCenters;
};

#endif