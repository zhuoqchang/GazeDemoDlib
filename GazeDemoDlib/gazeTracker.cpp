#include "gazeTracker.h"

#include <windows.h>
#include <sstream>
#include <iostream>

#include "readConfig.h"
#include "gazeDisplay.h"

#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Geometry>

typedef Eigen::Matrix<float, 3, Eigen::Dynamic> Vertices;
bool isClick;

std::vector<cv::Point2f> createCalibPoints13(int xmax, int ymax) {
	std::vector<cv::Point2f> p;

	float x1 = (float)xmax * 0.1;
	float x2 = (float)xmax * 0.5;
	float x3 = (float)xmax * 0.9;
	float x4 = (float)xmax * 0.3;
	float x5 = (float)xmax * 0.7;

	float y1 = (float)ymax * 0.1;
	float y2 = (float)ymax * 0.5;
	float y3 = (float)ymax * 0.9;
	float y4 = (float)ymax * 0.3;
	float y5 = (float)ymax * 0.7;

	p.push_back(cv::Point2f(x1, y1));
	p.push_back(cv::Point2f(x2, y1));
	p.push_back(cv::Point2f(x3, y1));
	p.push_back(cv::Point2f(x1, y2));
	p.push_back(cv::Point2f(x2, y2));
	p.push_back(cv::Point2f(x3, y2));
	p.push_back(cv::Point2f(x1, y3));
	p.push_back(cv::Point2f(x2, y3));
	p.push_back(cv::Point2f(x3, y3));
	p.push_back(cv::Point2f(x4, y4));
	p.push_back(cv::Point2f(x5, y4));
	p.push_back(cv::Point2f(x4, y5));
	p.push_back(cv::Point2f(x5, y5));

	return p;
}

void drawCircle(cv::Mat& img, cv::Point2f point) {
	cv::circle(img, point, 20, cv::Scalar(255, 255, 255), CV_FILLED, 8, 0);
}

bool withinRadius(cv::Point2i p1, cv::Point2i p2, int radius) {
	return cv::norm(p1 - p2) < radius;
}

void onMouse(int event, int x, int y, int flags, void* ptr) {
	if (event == cv::EVENT_LBUTTONDOWN || event == cv::EVENT_RBUTTONDOWN) {
		cv::Point* p = (cv::Point*)ptr;
		p->x = x;
		p->y = y;
		std::cout << "clicked point: " << x << ", " << y << std::endl;
		isClick = true;
	}
}

gazeTracker::gazeTracker() {

	loadConfigData();

	pixelWidth = GetSystemMetrics(SM_CXSCREEN);
	pixelHeight = GetSystemMetrics(SM_CYSCREEN);
	std::cout << pixelWidth << ", " << pixelHeight << std::endl;

	calibrationPoints = createCalibPoints13(pixelWidth, pixelHeight);
	// std::cout << "calibration points: " << calibrationPoints << std::endl;

	RC.init(frameWidth, frameHeight, frameRate);
	LT.init(frameWidth, frameHeight);
	FM.init(frameWidth, frameHeight);
	EM.init(numOfCalibPoints, samplePerCalibPoint);
	std::cout << "gaze tracker initialized " << std::endl;
}

gazeTracker::~gazeTracker(){
}

void gazeTracker::run() {
	RC.captureFrame();
	cout << "Could not open config file!" << endl;
	cv::Mat mappedColorFrame;
	Vertices pointCloud;
	RC.getMappedColorFrame(mappedColorFrame);
	cout << "Could not open config file!" << endl;
	RC.getPointCloud(pointCloud);
	cout << "Could not open config file!" << endl;
	LT.run(mappedColorFrame, pointCloud);
	cout << "Could not open config file!" << endl;
}

void gazeTracker::loadConfigData() {

	ifstream* infile = new ifstream;
	string filename = "..\\..\\data\\config.txt";
	infile->open(filename);
	if (!infile->is_open()) {
		cout << "Could not open config file!" << endl;
	}
	string line;

	getline(*infile, line, ':');
	getline(*infile, line);
	frameWidth = (int)readNumber(infile);

	getline(*infile, line, ':');
	getline(*infile, line);
	frameHeight = (int)readNumber(infile);

	getline(*infile, line, ':');
	getline(*infile, line);
	frameRate = (int)readNumber(infile);

	getline(*infile, line, ':');
	getline(*infile, line);
	numOfCalibPoints = (int)readNumber(infile);

	getline(*infile, line, ':');
	getline(*infile, line);
	samplePerCalibPoint = (int)readNumber(infile);

	getline(*infile, line, ':');
	getline(*infile, line);
	displayBlockH = (int)readNumber(infile);

	getline(*infile, line, ':');
	getline(*infile, line);
	displayBlockV = (int)readNumber(infile);

	cout << "frame width:" << endl << frameWidth << endl;
	cout << "frame height:" << endl << frameHeight << endl;
	cout << "frame rate:" << endl << frameRate << endl;
	cout << "number of calibration points:" << endl << numOfCalibPoints << endl;
	cout << "samples per calibration point:" << endl << samplePerCalibPoint << endl;
	cout << "number of horizontal display blocks:" << endl << displayBlockH << endl;
	cout << "number of vertical display blocks:" << endl << displayBlockV << endl;

	infile->close();
	delete infile;
}


void gazeTracker::getFrame(cv::Mat& frame) {
	RC.getMappedColorFrame(frame);
}

void gazeTracker::createFaceModel() {
	char key;

	cv::Mat display;
	cv::namedWindow("Face Model", cv::WINDOW_NORMAL);

	do {
		// omit first 30 frames 
		for (int i = 0; i < 30; i++) {
			run();

			cv::Mat frame;
			RC.getMappedColorFrame(frame);
			display = frame.clone();
			cv::flip(display, display, 1);
			cv::imshow("Face Model", display);
			cv::waitKey(5);
		}
		do {
			run();

			cv::Mat frame;
			RC.getMappedColorFrame(frame);
			display = frame.clone();
			cv::flip(display, display, 1);
			cv::imshow("Face Model", display);
			cv::waitKey(5);

		} while (LT.ifLmMissing());

		cv::Mat mappedColorFrame, mappedColorFace, lm2D;
		Vertices pointCloud;
		RC.getMappedColorFrame(mappedColorFrame);
		RC.getPointCloud(pointCloud);
		LT.getDepthLm(lm2D);
		FM.getFaceImage(mappedColorFrame, lm2D, mappedColorFace);

		//cv::Mat display = mappedColorFace.clone();
		//cv::flip(display, display, 1);
		//cv::imshow("Face Model", display);
		//cv::waitKey(5);
		//std::cout << "Is this face model OK? (y/n)" << std::endl;
		//std::cin >> key;
		break;
	} while (key != 'y');

	cv::destroyWindow("Face Model");
	// std::cout << "saving face model" << std::endl;
	// save the face model
	Vertices pointCloud, lm3D, rigidLm3D;
	cv::Mat lm2D, color;
	LT.getDepthLm(lm2D);
	LT.getWorldLm(lm3D);
	LT.getRigidLm(rigidLm3D);
	RC.getMappedColorFrame(color);
	RC.getPointCloud(pointCloud);
	FM.createReferenceFaceModel(lm2D, rigidLm3D, pointCloud);
	EM.createEyeModel(lm3D);

	// std::cout << "saved face model" << std::endl;
}

void gazeTracker::startCalibration() {
	// start calibration
	cv::namedWindow("Calibration", CV_WINDOW_NORMAL);
	cv::setWindowProperty("Calibration", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);

	//cv::namedWindow("leftEye", CV_WINDOW_NORMAL);
	//cv::namedWindow("rightEye", CV_WINDOW_NORMAL);
	//cv::resizeWindow("leftEye", 40, 24);
	//cv::resizeWindow("rightEye", 40, 24);

	std::string pathPrefix = "../../../data/calibPoint";
	std::string imgType = ".png";
	for (int i = 0; i < numOfCalibPoints; i++) {
		cv::Mat display = cv::Mat::zeros(cv::Size(pixelWidth, pixelHeight), CV_8UC3);
		drawCircle(display, calibrationPoints[i]);
		cv::imshow("Calibration", display);
		cv::waitKey(5);
		
		std::cout << "calibration point " << i+1 << ": " << calibrationPoints[i] << std::endl;

		// wait for mouse input
		cv::Point clickPoint = cv::Point2f(10000, 10000);
		cv::setMouseCallback("Calibration", onMouse, (void*)(&clickPoint));
		while (!withinRadius(clickPoint, calibrationPoints[i], 10)) {
			// wait
			run();
			cv::waitKey(10);
		}
		cv::setMouseCallback("Stimulus", NULL, NULL);

		// save next samplePerCalibPoint frames
		for (int j = 0; j < samplePerCalibPoint; j++) {
			run();
			cv::Mat lm2D, color;
			Vertices pointCloud, rigidLm3D;
			TransformType trans;
			LT.getRigidLm(rigidLm3D);
			//cv::Mat leftEyeImage, rightEyeImage;
			if (!LT.ifRigidLmMissing()) {
				RC.getPointCloud(pointCloud);
				RC.getMappedColorFrame(color);
				LT.getDepthLm(lm2D);
				FM.align(lm2D, rigidLm3D, pointCloud, trans);
				EM.calibrate(pointCloud, color, trans, clickPoint);
				//EM.getEyeImage(leftEyeImage, 0);
				//EM.getEyeImage(rightEyeImage, 1);
				//cv::imshow("leftEye", leftEyeImage);
				//cv::imshow("rightEye", rightEyeImage);
				//waitKey(5);
			}
			else {
				j--;
			}
		}
	}

	//cv::destroyWindow("leftEye");
	//cv::destroyWindow("rightEye");

	cv::destroyWindow("Calibration");
}

void gazeTracker::afterCalibration() {

	cv::namedWindow("leftEye", CV_WINDOW_NORMAL);
	cv::namedWindow("rightEye", CV_WINDOW_NORMAL);
	cv::resizeWindow("leftEye", 40, 24);
	cv::resizeWindow("rightEye", 40, 24);

	cv::Mat display;
	namedWindow("Gaze Test", WINDOW_NORMAL);
	setWindowProperty("Gaze Test", CV_WND_PROP_FULLSCREEN, 1);
	gazeDisplay GD(pixelWidth, pixelHeight, displayBlockH, displayBlockV);

	// wait for mouse input
	cv::Point clickPoint;
	cv::setMouseCallback("Gaze Test", onMouse, (void*)(&clickPoint));

	int samplePerClick = 3;
	isClick = false;

	int key = 0;
	while (key != 27) {
		if (key == 'p') {
			// change display type
			GD.changeType();
			key = 0;
		}
		
		if (isClick) {
			// save next samplePerClick frames
			for (int i = 0; i < samplePerClick; i++) {
				run();
				cv::Mat lm2D, color;
				cv::Mat leftEyeImage, rightEyeImage;
				Vertices pointCloud, rigidLm3D;
				TransformType trans;
				LT.getRigidLm(rigidLm3D);
				//cv::Mat leftEyeImage, rightEyeImage;
				if (!LT.ifRigidLmMissing()) {
					RC.getPointCloud(pointCloud);
					RC.getMappedColorFrame(color);
					LT.getDepthLm(lm2D);
					FM.align(lm2D, rigidLm3D, pointCloud, trans);
					EM.calibrate(pointCloud, color, trans, clickPoint);

					EM.getEyeImage(leftEyeImage, 0);
					EM.getEyeImage(rightEyeImage, 1);
					cv::imshow("leftEye", leftEyeImage);
					cv::imshow("rightEye", rightEyeImage);
					waitKey(5);

				}
				else {
					i--;
				}
			}
			isClick = false;
		}
		else {
			// display gaze estimation results
			run();
			cv::Mat lm2D, color;
			cv::Mat leftEyeImage, rightEyeImage;
			Vertices pointCloud, rigidLm3D;
			cv::Point leftFixPoint, rightFixPoint;
			TransformType trans;
			LT.getRigidLm(rigidLm3D);
			if (!LT.ifRigidLmMissing()) {
				RC.getPointCloud(pointCloud);
				RC.getMappedColorFrame(color);
				LT.getDepthLm(lm2D);
				FM.align(lm2D, rigidLm3D, pointCloud, trans);
				//std::cout << "rotation: " << std::endl << trans.first << std::endl;
				//std::cout << "translation: " << std::endl << trans.second << std::endl;
				EM.run(pointCloud, color, trans);
				EM.getEyeImage(leftEyeImage, 0);
				EM.getEyeImage(rightEyeImage, 1);
				cv::imshow("leftEye", leftEyeImage);
				cv::imshow("rightEye", rightEyeImage);
				EM.getFixPoint(leftFixPoint, 0);
				EM.getFixPoint(rightFixPoint, 1);
				display = GD.draw(leftFixPoint, rightFixPoint);
				key = cv::waitKey(5);
			}
			cv::imshow("Gaze Test", display);
		}
	}
	cv::destroyWindow("leftEye");
	cv::destroyWindow("rightEye");
	cv::destroyWindow("Gaze Test");
}