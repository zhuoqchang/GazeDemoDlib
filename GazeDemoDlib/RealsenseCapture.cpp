#include "RealsenseCapture.h"
#include <iostream>s

RealsenseCapture::RealsenseCapture() {
}


RealsenseCapture::~RealsenseCapture() {
	pxcSenseManager->Release();
}

void RealsenseCapture::init(int w, int h, float fr) {
	frameWidth = w;
	frameHeight = h;
	frameRate = fr;

	pxcVertices.resize(frameWidth * frameHeight);
	vertices.resize(3, frameWidth * frameHeight);

	pxcSenseManager = PXCSenseManager::CreateInstance();
	pxcSenseManager->EnableStream(PXCCapture::STREAM_TYPE_COLOR, frameWidth, frameHeight, frameRate);
	pxcSenseManager->EnableStream(PXCCapture::STREAM_TYPE_DEPTH, frameWidth, frameHeight, frameRate);
	pxcSenseManager->Init();

	pxcCaptureManager = pxcSenseManager->QueryCaptureManager();
	pxcCaptureDevice = pxcCaptureManager->QueryDevice();

	// set camera parameters
	pxcCaptureDevice->SetIVCAMAccuracy(PXCCapture::Device::IVCAM_ACCURACY_FINEST);
	pxcCaptureDevice->SetIVCAMMotionRangeTradeOff(9);
	pxcCaptureDevice->SetIVCAMLaserPower(16);
	pxcCaptureDevice->SetIVCAMFilterOption(4);
	pxcCaptureDevice->SetDepthConfidenceThreshold(1);

	projection = pxcCaptureDevice->CreateProjection();
}

void RealsenseCapture::captureFrame() {
	//Retrieve the color and depth samples aligned
	pxcSenseManager->AcquireFrame();
	PXCCapture::Sample *sample = pxcSenseManager->QuerySample();
	PXCImage::ImageData colorData;
	PXCImage::ImageData depthData;

	//Capture the color frame
	sample->color->AcquireAccess(PXCImage::ACCESS_READ, PXCImage::PIXEL_FORMAT_RGB24, &colorData);
	frameColor = cv::Mat(cv::Size(frameWidth, frameHeight), CV_8UC3, colorData.planes[0]).clone();

	//Capture the depth frame
	sample->depth->AcquireAccess(PXCImage::ACCESS_READ, PXCImage::PIXEL_FORMAT_DEPTH_F32, &depthData);
	frameDepth = cv::Mat(cv::Size(frameWidth, frameHeight), CV_32FC1, depthData.planes[0]).clone();

	//Compute point cloud 
	projection->QueryVertices(sample->depth, &pxcVertices[0]);
	pxcVertices2EigenMat();

	//Map color image to depth
	mapColorImageToDepthImage(sample->color, sample->depth);

	sample->color->ReleaseAccess(&colorData);
	sample->depth->ReleaseAccess(&depthData);
	pxcSenseManager->ReleaseFrame();
}

void RealsenseCapture::mapColorImageToDepthImage(PXCImage* color, PXCImage* depth) {
	PXCImage* mappedColor = projection->CreateColorImageMappedToDepth(depth, color);
	PXCImage::ImageData mappedColorData;
	mappedColor->AcquireAccess(PXCImage::ACCESS_READ, PXCImage::PIXEL_FORMAT_RGB24, &mappedColorData);
	frameMappedColor = cv::Mat(cv::Size(frameWidth, frameHeight), CV_8UC3, mappedColorData.planes[0]).clone();
	mappedColor->ReleaseAccess(&mappedColorData);
	mappedColor->Release();
}

// convert vertices to eigen matrix format
void RealsenseCapture::pxcVertices2EigenMat() {
	for (int i = 0; i < frameWidth * frameHeight; i++) {
		vertices(0, i) = pxcVertices[i].x;
		vertices(1, i) = pxcVertices[i].y;
		vertices(2, i) = pxcVertices[i].z;
	}
}

void RealsenseCapture::getColorFrame(cv::Mat& X) {
	X = frameColor;
}

void RealsenseCapture::getDepthFrame(cv::Mat& X) {
	X = frameDepth;
}

void RealsenseCapture::getMappedColorFrame(cv::Mat& X) {
	X = frameMappedColor;
}

void RealsenseCapture::getPointCloud(Vertices& X) {
	X = vertices;
}