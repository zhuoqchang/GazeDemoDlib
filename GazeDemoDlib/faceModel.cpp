#include "faceModel.h"

#include "geometricAlgorithms.h"
#include "icpPointToPlane.h"
#include "matrix.h"
#include <chrono>

typedef Eigen::Matrix<float, 3, Eigen::Dynamic> Vertices;

faceModel::faceModel() {
}


faceModel::~faceModel() {
	delete headModel;
}

void faceModel::init(int w, int h) {
	pointCloudRes.width = w;
	pointCloudRes.height = h;
}

void faceModel::createReferenceFaceModel(const cv::Mat& lm2D, const Vertices& rigidLm3D, const Vertices& pc) {
	//std::cout << "creating reference face model ..." << std::endl;
	createFaceModel(lm2D, rigidLm3D, pc);
	referenceRigidLm3D = currentRigidLm3D;
	referenceFacePointCloud = currentFacePointCloud;
	createIcpModel(referenceFacePointCloud);
	//std::cout << "finished creating reference face model ..." << std::endl;
}

void faceModel::align(const cv::Mat& lm2D, const Vertices& rigidLm3D, const Vertices& pc, TransformType& trans) {
	createFaceModel(lm2D, rigidLm3D, pc);
	alignFace(rigidLm3D);
	//applyTransform(pc, headPoseTrans, alignedPc);
	trans = headPoseTrans;
}

void faceModel::createFaceModel(const cv::Mat& lm2D, const Vertices& rigidLm3D, const Vertices& pc) {
	//std::cout << "creating face model ..." << std::endl;
	currentRigidLm3D = rigidLm3D;
	Vertices facePc;
	computeFacePointCloud(lm2D, pc, facePc);
	removeMissingData(facePc);
	//std::cout << "finished creating face model ..." << std::endl;
}

void faceModel::alignFace(const Vertices& rigidLm3D) {
	// compute rigid transform from rigid landmarks
	TransformType rigidTrans = getRigidTransformation(rigidLm3D, referenceRigidLm3D);
	Vertices facePcTrans;
	// apply rigid transform
	applyTransform(currentFacePointCloud, rigidTrans, facePcTrans);
	// compute rigid transform from icp
	//std::cout << "before icp transform" << std::endl;
	TransformType icpTrans = getIcpTransformation(facePcTrans);

	// apply icp transformation
	applyTransform(facePcTrans, icpTrans, alignedFacePointCloud);

	//alignedFacePointCloud = facePcTrans;
	headPoseTrans = std::make_pair(icpTrans.first * rigidTrans.first, icpTrans.first * rigidTrans.second + icpTrans.second);
	//headPoseTrans = rigidTrans;
}

void faceModel::createIcpModel(const Vertices& v) {
	int32_t dim = 3;
	int32_t s = (int32_t)v.cols();

	// allocate model and template memory
	double* M = (double*)calloc(3 * s, sizeof(double));

	// set model and template points
	for (int i = 0; i < s; i++) {
		M[i * 3 + 0] = (double)v(0, i);
		M[i * 3 + 1] = (double)v(1, i);
		M[i * 3 + 2] = (double)v(2, i);
	}

	headModel = new IcpPointToPlane(M, s, dim);
}

void faceModel::computeFacePointCloud(const cv::Mat& lm2D, const Vertices& pc, Vertices& facePc) {
	rect r = getFaceRect(lm2D);

	facePc.resize(3, r.h * r.w);
	// copy each row of point cloud within the face rect into facePc 
	for (int i = 0; i < r.h; i++) {
		int idx = (r.y + i - 1) * pointCloudRes.width + r.x;
		facePc.block(0, r.w * i, 3, r.w) = pc.block(0, idx, 3, r.w);
	}
}

rect faceModel::getFaceRect(const cv::Mat& lm) {
	rect r;
	r.x = (int)lm.at<float>(0, 20 - 1);
	r.y = (int)std::min(lm.at<float>(1, 5 - 1), lm.at<float>(1, 6 - 1));
	r.w = (int)(lm.at<float>(0, 29 - 1) - r.x);
	r.h = (int)(lm.at<float>(1, 17 - 1) - r.y);

	r.x = r.x + 0.1 * r.w;
	r.w = 0.8 * r.w;
	r.y = r.y + 0.2 * r.h;
	r.h = 0.8 * r.h;
	return r;
}

void faceModel::cropFacePointCloud(rect r, const Vertices& pc, Vertices& facePc) {
	facePc.resize(3, r.h * r.w);
	// copy each row of point cloud within the face rect into facePc 
	for (int i = 0; i < r.h; i++) {
		int idx = (r.y + i - 1) * pointCloudRes.width + r.x;
		facePc.block(0, r.w * (i - 1), 3, r.w) = pc.block(0, idx, 3, r.w);
	}
}

void faceModel::getFaceImage(const cv::Mat& color, const cv::Mat& lm2D, cv::Mat& img) {
	rect r = getFaceRect(lm2D);
	img = color(cv::Rect(r.x, r.y, r.w, r.h)).clone();
}

TransformType faceModel::getRigidTransformation(const Vertices& src, const Vertices& dst) {
	PointsType p1s, p2s;
	p1s.resize(numOfRigidLm);
	for (int i = 0; i < numOfRigidLm; ++i) {
		p1s[i][0] = src(0, i);
		p1s[i][1] = src(1, i);
		p1s[i][2] = src(2, i);
	}
	p2s.resize(numOfRigidLm);
	for (int i = 0; i < numOfRigidLm; ++i) {
		p2s[i][0] = dst(0, i);
		p2s[i][1] = dst(1, i);
		p2s[i][2] = dst(2, i);
	}

	TransformType trans;

	//cout << "computing the rigid transformations...\n";
	trans = computeRigidTransform(p1s, p2s);
	//std::cout << trans.first << endl;
	//std::cout << (trans.second)[0] << "  " << (trans.second)[1] << "  " << (trans.second)[2] << endl;
	//cout << endl;
	return trans;
}

TransformType faceModel::getIcpTransformation(const Vertices& v) {

	int32_t dim = 3;
	int32_t s = (int32_t)v.cols();

	// allocate model and template memory
	double* M = (double*)calloc(3 * s, sizeof(double));

	// set model and template points
	for (int i = 0; i < s; i++) {
		M[i * 3 + 0] = (double)v(0, i);
		M[i * 3 + 1] = (double)v(1, i);
		M[i * 3 + 2] = (double)v(2, i);
	}
	// start with identity as initial transformation
	// in practice you might want to use some kind of prediction here
	Mat::Matrix R = Mat::Matrix::eye(3);
	Mat::Matrix t(3, 1);

	// run point-to-plane ICP (-1 = no outlier threshold)
	auto t4 = std::chrono::system_clock::now();
	headModel->fit(M, s, R, t, -1);

	auto t5 = std::chrono::system_clock::now();
	//std::cout << "icp creation took: "
	//	<< std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()
	//	<< " milliseconds\n" << std::endl;
	//std::cout << "icp iteration took: "
	//	<< std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count()
	//	<< " milliseconds\n" << std::endl;

	Eigen::Matrix3d RR;
	Eigen::Vector3d tt;

	for (int32_t i = 0; i < 3; i++) {
		for (int32_t j = 0; j < 3; j++) {
			R.getData(&RR(i, j), i, j, i, j);
		}
	}
	
	for (int32_t i = 0; i < 3; i++) {
		t.getData(&tt(i), 0, i, 0, i);
	}

	// results
	//std::cout << std::endl << "Transformation results:" << std::endl;
	//std::cout << "R:" << std::endl << RR << std::endl << std::endl;
	//std::cout << "t:" << std::endl << tt << std::endl << std::endl;

	// free memory
	free(M);

	// success
	return std::make_pair(RR, tt);
}

void faceModel::removeMissingData(const Vertices& pc) {
	int n = pc.cols();
	int numMissing = 0;
	// check number of missing values
	for (int i = 0; i < n; i++) {
		if (pc(2, i) < 50.0f) {	
			numMissing++;
		}
	}
	//std::cout << "missing data points: " << numMissing << std::endl;
	currentFacePointCloud.resize(3, n - numMissing);
	int count = 0;
	for (int i = 0; i < n; i++) {
		if (pc(2, i) > 50.0f) {
			currentFacePointCloud.col(count) = pc.col(i);
			count++;
		}
	}
}