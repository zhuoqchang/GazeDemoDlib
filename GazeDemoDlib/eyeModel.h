#pragma once

#ifndef EYEMODEL_H
#define EYEMODEL_H

#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>		
#include <opencv2/imgproc/imgproc.hpp>

typedef std::pair<Eigen::Matrix3d, Eigen::Vector3d> TransformType;
typedef Eigen::Matrix<float, 3, Eigen::Dynamic> Vertices;

struct params {
	Eigen::Matrix3f Tpix2cam;
	Eigen::Matrix3f Tcam2pix;
	Eigen::Vector4f monitorPlane;
};

struct eyeRegion {
	Eigen::Vector3f center;
	float width;
	float height;
};

struct eyeRes {
	int width;
	int height;
};

struct colorPointCloud {
	Vertices pc;
	cv::Mat tex;
	colorPointCloud() {};
	colorPointCloud(Vertices p, cv::Mat t) {
		pc = p;
		tex = t;
	}
};

class eyeModel {
public:
	eyeModel();
	~eyeModel();

	void init(int a, int b);
	void createEyeModel(const Vertices& lm3D);
	void calibrate(const Vertices& pc, const cv::Mat& color, const TransformType& trans, const cv::Point& pt);
	void run(const Vertices& pc, const cv::Mat& tex, const TransformType& trans);
	void getEyeImage(cv::Mat& img, int id);
	void getFixPoint(cv::Point& p, int id);

private:
	params calibParams;

	eyeRegion leftEyeRegion;
	eyeRegion rightEyeRegion;
	eyeRes eyeRegionRes;

	int numOfCalibPoints, samplesPerCalibPoint;
	cv::Mat leftEyeImg, rightEyeImg;
	cv::Mat leftEyeImgRec, rightEyeImgRec;
	cv::Mat leftEyeDictionary, rightEyeDictionary;
	Vertices leftGazeVector, rightGazeVector;
	Vertices leftGazeDictionary, rightGazeDictionary;
	cv::Point leftFixPoint, rightFixPoint;
	int dcount, maxDCount;

	void loadConfigData();
	void computeTexture(const cv::Mat& color, cv::Mat& tex);
	void removeMissingData(const colorPointCloud& in, colorPointCloud& out);
	void getEyeRegion(const Eigen::MatrixXf& eyeLm3D, eyeRegion& eyeReg);
	void createEyeImage(const colorPointCloud& colorPc, const eyeRegion& eye, cv::Mat& eyeImg);
	void cropEye(const colorPointCloud& cpc, const eyeRegion& eye, colorPointCloud& cpcCropped, cv::Rect& rect);
	void interpolateEyeImage(const colorPointCloud& cpc, const eyeRegion& eye, const cv::Rect& r, cv::Mat& eyeImg);
	void runOmp();
	cv::Point computeFixPoint(const Vertices& gazeVec, const Vertices& eyeCenter, const TransformType& trans);
	cv::Point coordToPixel(const Vertices& pt3);
	Vertices pixelToCoord(const cv::Point& pt2);
};

#endif