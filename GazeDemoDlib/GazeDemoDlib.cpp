#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>		
#include <opencv2/imgproc/imgproc.hpp>

#ifdef OPENCV_VERSION_GREATER_THAN300
#include <opencv2/videoio/videoio_c.h>
#include <opencv2/videoio/videoio.hpp>
#endif

#include "gazeTracker.h"

#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <chrono>

using namespace std;

//! This function draws the miliseconds per frame.
/*!
\param img  Input image where the time per frame is shown.
\param ms  the value to be shown.
*/
inline void drawTimePerFrame(cv::Mat& img, chrono::milliseconds ms) {
	// variable containing the string
	char frameRate[50];
	// where to display frame rate
	cv::Point frLoc(100, 20);
	// where to display frame rate shadow
	cv::Point frLocS(101, 21);
	// text font for displaying frame rate
	int frFont = cv::FONT_HERSHEY_PLAIN;
	double fontScale = 1.5;
	sprintf(frameRate, "%02d ms/Frame", (unsigned int)ms.count());
	cv::putText(img, frameRate, frLocS, frFont, fontScale, cv::Scalar(100, 100, 100, 0.3), 1);
	cv::putText(img, frameRate, frLoc, frFont, fontScale, cv::Scalar(50, 50, 255), 1);
}

int main(int argc, char **argv) {

	cv::namedWindow("Color", cv::WINDOW_NORMAL);

	// Initialize GazeTracker
	cout << "initialize gaze tracker ... " << endl;
	gazeTracker gt;
	cout << "finished initializing gaze tracker ... " << endl;

	chrono::time_point<std::chrono::system_clock> start, end;

	// Variable that stores the value of the keys pressed
	int key = 0;

	// Local images for display
	cv::Mat displayColor;
	cv::Mat displayDepth;

	// Press Esc to quit
	while (key != 27 && key != 'p') {

		// Start measuring time
		start = chrono::system_clock::now();

		// Get frames
		gt.run();

		cv::Mat frame;
		gt.getFrame(frame);
		frame.copyTo(displayColor);

		cv::flip(displayColor, displayColor, 1);

		// Measure and plot the iteration time
		end = chrono::system_clock::now();
		chrono::milliseconds ms = chrono::duration_cast<std::chrono::milliseconds>(end - start);
		drawTimePerFrame(displayColor, ms);

		// Showing the current frame
		cv::imshow("Color", displayColor);
		// cv::imshow("Depth", displayDepth);

		key = cv::waitKey(5);
	}
	if (key == 'p') {
		cv::destroyWindow("Color");
		// create face model
		// std::cout << "Creating face model, please face frontal and hold still." << endl;
		gt.createFaceModel();
		// std::cout << "Calibrate eyes, please fix your eyes on the black dots." << endl;
		gt.startCalibration();
		gt.afterCalibration();
	}
	std::cout << "return success" << endl;
	return 0;
}



