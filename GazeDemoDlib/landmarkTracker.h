#pragma once

#ifndef LANDMARKTRACKER_H
#define LANDMARKTRACKER_H

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>		
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "dlibFaceDetector.h"

#define NUM_OF_LANDMARKS 51

const int rigidNum = 9;
const int borderNum = 5;
const int rigidIdx[rigidNum] = { 11, 14, 15, 17, 19, 20, 23, 26, 29 };
const int borderIdx[borderNum] = { 5, 6, 17, 20, 29 };
const float thresh = 50.0f;

typedef Eigen::Matrix<float, 3, Eigen::Dynamic> Vertices;

class landmarkTracker{
public:
	landmarkTracker();
	~landmarkTracker();

	void init(int w, int h);
	void run(cv::Mat& color, const Vertices& vertices);

	bool ifLmMissing();
	bool ifRigidLmMissing();
	bool ifBorderLmMissing();

	void getDepthLm(cv::Mat& X);
	void getWorldLm(Vertices& X);
	void getRigidLm(Vertices& X);

private:

	dlibFaceDetector dlibFace;

	cv::Mat depthLm;
	Vertices worldLm;
	Vertices rigidLm;
	
	int height, width;

	bool isDetect;
	bool isMissing;
	bool isRigidMissing;
	bool isBorderMissing;

	void check();
	void checkRigid();
	void checkBorder();

	void detect(cv::Mat& color);
	void depthToWorld(const Vertices& vertices);
	void computeRigidLm();
};

#endif