#pragma once

#ifndef DLIBFACEDETECTOR_H
#define DLIBFACEDETECTOR_H

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>		
#include <opencv2/imgproc/imgproc.hpp>

using namespace dlib;
using namespace cv;

class dlibFaceDetector {
public:
	dlibFaceDetector();
	~dlibFaceDetector();

	bool detectLandmarks(const cv::Mat& img, cv::Mat& lm);

private:
	frontal_face_detector detector;
	shape_predictor pose_model;
	std::vector<dlib::rectangle> faces;
	int count;
};


#endif