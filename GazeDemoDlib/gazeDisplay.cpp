#include "gazeDisplay.h"
#include <iostream>

double pointDist(const cv::Point2d& p1, const cv::Point2d& p2) {
	return cv::norm(p1 - p2);
}

gazeDisplay::gazeDisplay(int w, int h, int bh, int bv) {
	cv::namedWindow("Gaze Test", CV_WINDOW_NORMAL);
	cv::setWindowProperty("Gaze Test", CV_WND_PROP_FULLSCREEN, 1);

	width = w;
	height = h;

	hblocks = bh;
	vblocks = bv;

	for (int i = 0; i < vblocks; i++) {
		for (int j = 0; j < hblocks; j++) {
			regionCenters.push_back(cv::Point2d(w / (2 * hblocks) * (2 * j + 1), h / (2 * vblocks) * (2 * i + 1)));
		}
	}

	int nblocks = vblocks * hblocks;

	darkColors.push_back(cv::Scalar(102, 0, 0));
	darkColors.push_back(cv::Scalar(102, 51, 0));
	darkColors.push_back(cv::Scalar(102, 102, 0));
	darkColors.push_back(cv::Scalar(51, 102, 0));
	darkColors.push_back(cv::Scalar(0, 102, 0));
	darkColors.push_back(cv::Scalar(0, 102, 51));
	darkColors.push_back(cv::Scalar(0, 102, 102));
	darkColors.push_back(cv::Scalar(0, 51, 102));
	darkColors.push_back(cv::Scalar(0, 0, 102));
	darkColors.push_back(cv::Scalar(51, 0, 102));
	darkColors.push_back(cv::Scalar(102, 0, 102));
	darkColors.push_back(cv::Scalar(102, 0, 51));
	darkColors.push_back(cv::Scalar(32, 32, 32));

	brightColors.push_back(cv::Scalar(255, 0, 0));
	brightColors.push_back(cv::Scalar(255, 128, 0));
	brightColors.push_back(cv::Scalar(255, 255, 0));
	brightColors.push_back(cv::Scalar(128, 255, 0));
	brightColors.push_back(cv::Scalar(0, 255, 0));
	brightColors.push_back(cv::Scalar(0, 255, 128));
	brightColors.push_back(cv::Scalar(0, 255, 255));
	brightColors.push_back(cv::Scalar(0, 128, 255));
	brightColors.push_back(cv::Scalar(0, 0, 255));
	brightColors.push_back(cv::Scalar(128, 0, 255));
	brightColors.push_back(cv::Scalar(255, 0, 255));
	brightColors.push_back(cv::Scalar(255, 0, 128));
	brightColors.push_back(cv::Scalar(128, 128, 128));

	type = 0;

	int colorCounter = 0;
	std::vector<cv::Mat> blocks;
	
	// create blocks
	for (int i = 0; i < nblocks; i++) {
		blocks.push_back(cv::Mat(h / vblocks, w / hblocks, CV_8UC3, darkColors.at(colorCounter % darkColors.size())));
		colorCounter++;
	}
	// concat blocks horizontally
	std::vector<cv::Mat> hconcatBlocks;
	for (int i = 0; i < vblocks; i++) {
		cv::Mat temp;

		std::vector<cv::Mat>::const_iterator first = blocks.begin() + hblocks * i;
		std::vector<cv::Mat>::const_iterator last = blocks.begin() + hblocks * (i + 1);
		std::vector<cv::Mat> newVec(first, last);

		cv::hconcat(newVec, temp);
		hconcatBlocks.push_back(temp);
	}
	// concat blocks vertically
	vconcat(hconcatBlocks, bg0);

	bg1 = cv::Mat::zeros(h, w, CV_8UC3);

	std::cout << "gaze display initialized " << std::endl;
}


gazeDisplay::~gazeDisplay() {
	cv::destroyWindow("Gaze Test");
}

void gazeDisplay::changeType() {
	type = !type;
}

cv::Mat gazeDisplay::draw(const cv::Point& p1, const cv::Point& p2) {

	cv::Mat display;
	if (type) {
		display = bg1.clone();
		//cv::circle(display, p1, 5.0, cv::Scalar(0, 0, 255), -1, 8);
		//cv::circle(display, p2, 5.0, cv::Scalar(255, 0, 0), -1, 8);
		cv::Point p = 0.5 * (p1 + p2); // compute middle point
		cv::circle(display, p, 5.0, cv::Scalar(0, 255, 0), -1, 8);
	}
	else {
		display = bg0.clone();
		std::vector<double> dist;
		cv::Point p = 0.5 * (p1 + p2); // compute middle point
		// compute distance to each region center
		if (p.x > 0 && p.x < width && p.y > 0 && p.y < height) {
			for (int i = 0; i < hblocks * vblocks; i++) {
				dist.push_back(pointDist(p, regionCenters.at(i)));
			}
			int idx = std::min_element(dist.begin(), dist.end()) - dist.begin();

			int row = idx / hblocks;
			int col = idx % hblocks;

			display(cv::Rect(width * col / hblocks, height * row / vblocks, width / hblocks, height / vblocks)).setTo(brightColors.at(idx % brightColors.size()));
		}
	}

	return display;
}