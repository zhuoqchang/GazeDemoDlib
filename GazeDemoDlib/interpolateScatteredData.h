#pragma once

#include <iostream>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>		
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

float getDist(const cv::Point2f& p1, const cv::Point2f& p2) {
	return pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2);
}

bool interpolateScatteredData(cv::Subdiv2D& subdiv, const cv::Point2f& point, const std::vector<uchar>& vals, uchar& ival) {
	int e0, e1, v, v0, v1, v2;
	int status = subdiv.locate(point, e0, v);
	if (status == CV_PTLOC_INSIDE) {
		// get next edge
		e1 = subdiv.getEdge(e0, CV_NEXT_AROUND_LEFT);

		// get vertices of edges
		v0 = subdiv.edgeOrg(e0);
		v1 = subdiv.edgeDst(e0);
		v2 = subdiv.edgeDst(e1);

		// get points
		cv::Point2f p0 = subdiv.getVertex(v0);
		cv::Point2f p1 = subdiv.getVertex(v1);
		cv::Point2f p2 = subdiv.getVertex(v2);

		float d0 = 1.0f / getDist(point, p0);
		float d1 = 1.0f / getDist(point, p1);
		float d2 = 1.0f / getDist(point, p2);
		float d = d0 + d1 + d2;
		ival = (uchar)(float(vals[v0]) * d0 / d + (float)vals[v1] * d1 / d + (float)vals[v2] * d2 / d);
	}
	else if (status == CV_PTLOC_ON_EDGE) {
		// get vertices of edges
		v0 = subdiv.edgeOrg(e0);
		v1 = subdiv.edgeDst(e0);

		// get points
		cv::Point2f p0 = subdiv.getVertex(v0);
		cv::Point2f p1 = subdiv.getVertex(v1);

		float d0 = 1.0f / getDist(point, p0);
		float d1 = 1.0f / getDist(point, p1);
		float d = d0 + d1;
		ival = (uchar)(float(vals[v0]) * d0 / d + (float)vals[v1] * d1 / d );
	}
	else if (status == CV_PTLOC_VERTEX) {
		// get points
		cv::Point2f p = subdiv.getVertex(v);
		ival = vals[v];
	}
	else {
		std::cout << "Test point error!" << std::endl;
		return false;
	}
	return true;
}