#include "dlibFaceDetector.h"

#define FACE_DOWNSAMPLE_RATIO 2
#define SKIP_FRAMES 2
#define NUM_OF_LANDMARKS 51

dlibFaceDetector::dlibFaceDetector() {
	detector = get_frontal_face_detector();
	deserialize("..\\..\\data\\shape_predictor_68_face_landmarks.dat") >> pose_model;
	count = 0;
}


dlibFaceDetector::~dlibFaceDetector() {

}

bool dlibFaceDetector::detectLandmarks(const cv::Mat& im, cv::Mat& lm) {
	// Resize image for face detection
	cv::Mat im_small;
	cv::resize(im, im_small, cv::Size(), 1.0 / FACE_DOWNSAMPLE_RATIO, 1.0 / FACE_DOWNSAMPLE_RATIO);

	// Change to dlib's image format. No memory is copied.
	cv_image<bgr_pixel> cimg_small(im_small);
	cv_image<bgr_pixel> cimg(im);

	// Detect faces on resize image
	std::vector<dlib::rectangle> faces;
	if (count % SKIP_FRAMES == 0) {
		faces = detector(cimg_small);
	}

	// Find the pose of each face.
	cv::Mat lmx = cv::Mat::zeros(2, NUM_OF_LANDMARKS, CV_32F);
	if (faces.size() > 0) {
		// Resize obtained rectangle for full resolution image. 
		dlib::rectangle r(
			(long)(faces[0].left() * FACE_DOWNSAMPLE_RATIO),
			(long)(faces[0].top() * FACE_DOWNSAMPLE_RATIO),
			(long)(faces[0].right() * FACE_DOWNSAMPLE_RATIO),
			(long)(faces[0].bottom() * FACE_DOWNSAMPLE_RATIO)
		);

		// Landmark detection on full sized image
		full_object_detection shape = pose_model(cimg, r);

		for (int i = 0; i < NUM_OF_LANDMARKS; ++i) {
			lmx.at<float>(0, i) = shape.part(i+17).x();
			lmx.at<float>(1, i) = shape.part(i+17).y();
		}
		lmx.copyTo(lm);
		return true;
	}
	else {
		return false;
	}
}