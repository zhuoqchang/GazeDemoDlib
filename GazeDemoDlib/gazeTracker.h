#pragma once

#ifndef GAZETRACKER_H
#define GAZETRACKER_H

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>		
#include <opencv2/imgproc/imgproc.hpp>

#include "faceModel.h"
#include "RealsenseCapture.h"
#include "landmarkTracker.h"
#include "eyeModel.h"

using namespace std;
typedef std::pair<Eigen::Matrix3d, Eigen::Vector3d> TransformType;

class gazeTracker {
public:
	gazeTracker();
	~gazeTracker();

	void run();
	void getFrame(cv::Mat& frame);
	void createFaceModel();
	void startCalibration();
	void afterCalibration();

private:

	void loadConfigData();
	// face model data
	faceModel FM;
	eyeModel EM;
	RealsenseCapture RC;
	landmarkTracker LT;

	int frameHeight, frameWidth, frameRate;
	int numOfCalibPoints, samplePerCalibPoint;
	std::vector<cv::Point2f> calibrationPoints;
	int pixelWidth, pixelHeight;
	int displayBlockH, displayBlockV;
};

#endif