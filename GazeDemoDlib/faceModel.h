#pragma once

#ifndef FACEMODEL_H
#define FACEMODEL_H

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>		
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Geometry>

const int numOfLm = 49;
const int numOfRigidLm = 9;
const int numOfBorderLm = 5;
const int RigidLmIdx[numOfRigidLm] = { 11, 14, 15, 17, 19, 20, 23, 26, 29 };
const int BorderLmIdx[numOfBorderLm] = { 5, 6, 17, 20, 29 };

struct res {
	int width;
	int height;
};

struct rect {
	int x;
	int y;
	int w;
	int h;
};

typedef std::pair<Eigen::Matrix3d, Eigen::Vector3d> TransformType;
typedef std::vector<Eigen::Vector3d> PointsType;
typedef Eigen::Matrix<float, 3, Eigen::Dynamic> Vertices;

class IcpPointToPlane;

class faceModel {
public:
	faceModel();
	~faceModel();
	void init(int w, int h);
	void align(const cv::Mat& depthLm2D, const Vertices& rigidLm3D, const Vertices& pc, TransformType& trans);
	void createReferenceFaceModel(const cv::Mat& depthLm2D, const Vertices& rigidLm3D, const Vertices& pc);
	void getFaceImage(const cv::Mat& color, const cv::Mat& lm2D, cv::Mat& img); // does not reshape

private:
	Vertices currentRigidLm3D, referenceRigidLm3D;
	Vertices currentFacePointCloud, referenceFacePointCloud, alignedFacePointCloud;
	TransformType headPoseTrans;
	IcpPointToPlane* headModel;

	res pointCloudRes;

	void createFaceModel(const cv::Mat& depthLm2D, const Vertices& rigidLm3D, const Vertices& pc);
	rect getFaceRect(const cv::Mat& lm2D);
	void cropFacePointCloud(rect r, const Vertices& pc, Vertices& facePc);
	void computeFacePointCloud(const cv::Mat& depthLm2D, const Vertices& pc, Vertices& facePc);

	void alignFace(const Vertices& lm3D);

	void createIcpModel(const Vertices& pc);
	TransformType getRigidTransformation(const Vertices& src, const Vertices& dst);
	TransformType getIcpTransformation(const Vertices& v);
	void removeMissingData(const Vertices& pc);
};

#endif