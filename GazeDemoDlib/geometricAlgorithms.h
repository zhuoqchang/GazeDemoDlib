#pragma once

#ifndef GEOMETRICALGORITHMS_H
#define GEOMETRICALGORITHMS_H

#include <iostream>
#include <ctime>

#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Geometry>

typedef std::pair<Eigen::Matrix3d, Eigen::Vector3d> TransformType;
typedef std::vector<Eigen::Vector3d> PointsType;
typedef Eigen::Matrix<float, 3, Eigen::Dynamic> Vertices;

TransformType computeRigidTransform(const PointsType& src, const PointsType& dst);

Vertices computeNormalizedVector(const Vertices& p1, const Vertices& p2);

Vertices computePointPlaneIntersection(const Vertices& point, const Vertices& dir, const Eigen::Vector4f& plane);

TransformType computeInvTransform(const TransformType& trans);

void applyTransform(const Vertices& src, const TransformType trans, Vertices& dst);

void applyInvTransform(const Vertices& src, const TransformType trans, Vertices& dst);

#endif