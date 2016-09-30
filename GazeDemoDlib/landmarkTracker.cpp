#include "landmarkTracker.h"

landmarkTracker::landmarkTracker() {
}

landmarkTracker::~landmarkTracker() {
}

void landmarkTracker::init(int w, int h) {
	width = w;
	height = h;
	isDetect = false;
	worldLm.resize(Eigen::NoChange, NUM_OF_LANDMARKS);
	rigidLm.resize(Eigen::NoChange, rigidNum);
}

void landmarkTracker::run(cv::Mat& color, const Vertices& vertices) {
	detect(color);
	if (isDetect) {
		depthToWorld(vertices);
		computeRigidLm();
		check();
		checkRigid();
		checkBorder();
	}
	else {
		isMissing = true;
		isBorderMissing = true;
		isRigidMissing = true;
	}
}

void landmarkTracker::detect(cv::Mat& color) {
	isDetect = dlibFace.detectLandmarks(color, depthLm);
	if (isDetect) {
			std::cout << "(" << depthLm.at<float>(0, 14) << ", " << depthLm.at<float>(1, 14) << ")" << std::endl;
	}
}

void landmarkTracker::depthToWorld(const Vertices& vertices) {
	for (int i = 0; i < NUM_OF_LANDMARKS; i++) {
		worldLm(0, i) = vertices(0, (int)depthLm.at<float>(1, i) * width + (int)depthLm.at<float>(0, i));
		worldLm(1, i) = vertices(1, (int)depthLm.at<float>(1, i) * width + (int)depthLm.at<float>(0, i));
		worldLm(2, i) = vertices(2, (int)depthLm.at<float>(1, i) * width + (int)depthLm.at<float>(0, i));
	}
}

void landmarkTracker::computeRigidLm() {
	for (int i = 0; i < rigidNum; i++) {
		rigidLm.col(i) = worldLm.col(rigidIdx[i] - 1);
	}
}

void landmarkTracker::check() {
	isMissing = false;
	for (int i = 0; i < NUM_OF_LANDMARKS; i++) {
		if (worldLm(2, i) < thresh) {
			isMissing = true;
			break;
		}
	}
}

void landmarkTracker::checkRigid() {
	isRigidMissing = false;
	for (int i = 0; i < rigidNum; i++) {
		if (worldLm(2, rigidIdx[i] - 1) < thresh) {
			isRigidMissing = true;
			break;
		}
	}
}

void landmarkTracker::checkBorder() {
	isBorderMissing = false;
	for (int i = 0; i < borderNum; i++) {
		if (worldLm(2, borderIdx[i] - 1) < thresh) {
			isBorderMissing = true;
			break;
		}
	}
}

bool landmarkTracker::ifLmMissing() {
	return isMissing;
}

bool landmarkTracker::ifRigidLmMissing() {
	return isRigidMissing;
}

bool landmarkTracker::ifBorderLmMissing() {
	return isBorderMissing;
}

void landmarkTracker::getDepthLm(cv::Mat& X) {
	X = depthLm;
}

void landmarkTracker::getWorldLm(Vertices& X) {
	X = worldLm;
}

void landmarkTracker::getRigidLm(Vertices& X) {
	X = rigidLm;
}