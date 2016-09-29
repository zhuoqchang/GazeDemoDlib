#include "geometricAlgorithms.h"

#include <iostream>
#include <ctime>

TransformType computeRigidTransform(const PointsType& src, const PointsType& dst) {
	assert(src.size() == dst.size());
	int pairSize = src.size();
	Eigen::Vector3d center_src(0, 0, 0), center_dst(0, 0, 0);
	for (int i = 0; i < pairSize; ++i) {
		center_src += src[i];
		center_dst += dst[i];
	}
	center_src /= (double)pairSize;
	center_dst /= (double)pairSize;

	Eigen::MatrixXd S(pairSize, 3), D(pairSize, 3);
	for (int i = 0; i<pairSize; ++i) {
		for (int j = 0; j < 3; ++j)
			S(i, j) = src[i][j] - center_src[j];
		for (int j = 0; j < 3; ++j)
			D(i, j) = dst[i][j] - center_dst[j];
	}
	Eigen::MatrixXd Dt = D.transpose();
	Eigen::Matrix3d H = Dt*S;
	Eigen::Matrix3d W, U, V;

	Eigen::JacobiSVD<Eigen::MatrixXd> svd;
	Eigen::MatrixXd H_(3, 3);
	for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) H_(i, j) = H(i, j);
	svd.compute(H_, Eigen::ComputeThinU | Eigen::ComputeThinV);
	if (!svd.computeU() || !svd.computeV()) {
		std::cerr << "decomposition error" << std::endl;
		return std::make_pair(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
	}
	Eigen::Matrix3d Vt = svd.matrixV().transpose();
	Eigen::Matrix3d R = svd.matrixU()*Vt;
	Eigen::Vector3d t = center_dst - R*center_src;

	return std::make_pair(R, t);
}

Vertices computeNormalizedVector(const Vertices& p1, const Vertices& p2) {
	Vertices p = p2 - p1;
	return p / p.norm();
}

Vertices computePointPlaneIntersection(const Vertices& point, const Vertices& dir, const Eigen::Vector4f& plane) {
	float lhs = (plane.head(3).transpose() * point)(0) + plane(3);
	float rhs = -(plane.head(3).transpose() * dir)(0);
	float alpha = lhs / rhs;
	Vertices fp = point + alpha * dir;
	return fp;
}

TransformType computeInvTransform(const TransformType& trans) {
	Eigen::Matrix3d R = trans.first.transpose();
	Eigen::Vector3d t = -trans.first.transpose() * trans.second;
	return std::make_pair(R, t);
}

void applyTransform(const Vertices& src, const TransformType trans, Vertices& dst) {
	dst.resize(src.rows(), src.cols());
	dst = (trans.first.cast<float>() * src).colwise() + trans.second.cast<float>();
}

void applyInvTransform(const Vertices& src, const TransformType trans, Vertices& dst) {
	TransformType invTrans = computeInvTransform(trans);
	dst.resize(src.rows(), src.cols());
	dst = (invTrans.first.cast<float>() * src).colwise() + invTrans.second.cast<float>();
}