#pragma once

#ifndef REALSENSECAPTURE_H
#define REALSENSECAPTURE_H

#include <pxcsensemanager.h>
#include <pxcprojection.h>

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>		
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

typedef Eigen::Matrix<float, 3, Eigen::Dynamic> Vertices;

class RealsenseCapture {
public:
	RealsenseCapture();
	~RealsenseCapture();
	
	void init(int w, int h, float fr);
	void captureFrame();
	void getColorFrame(cv::Mat& X);
	void getDepthFrame(cv::Mat& X);
	void getMappedColorFrame(cv::Mat& X);
	void getPointCloud(Vertices& X);

private:
	int frameWidth;
	int frameHeight;
	int frameRate;

	cv::Mat frameColor;
	cv::Mat frameDepth;
	cv::Mat frameMappedColor;

	std::vector<PXCPoint3DF32> pxcVertices;
	Vertices vertices;

	PXCSenseManager* pxcSenseManager;
	PXCCaptureManager* pxcCaptureManager;
	PXCCapture::Device* pxcCaptureDevice;
	PXCProjection* projection;

	void mapColorImageToDepthImage(PXCImage* color, PXCImage* depth);
	void pxcVertices2EigenMat();
};

#endif