#include "eyeModel.h"
#include <iostream>
#include "interpolateScatteredData.h"
#include "omp.h"
#include "geometricAlgorithms.h"
#include "readConfig.h"

std::vector<float> linspace(float a, float b, int n) {
	std::vector<float> array;
	float step = (b - a) / (n - 1);
	while (a <= b) {
		array.push_back(a);
		a += step;           // could recode to better handle rounding errors
	}
	return array;
}

eyeModel::eyeModel() {
}

eyeModel::~eyeModel() {
}

void eyeModel::init(int a, int b) {
	// cout << "start eye init" << endl;
	eyeRegionRes.width = 40;
	eyeRegionRes.height = 24;

	numOfCalibPoints = a;
	samplesPerCalibPoint = b;

	//int n = numOfCalibPoints * samplesPerCalibPoint;
	maxDCount = 1000;
	leftEyeDictionary = cv::Mat_<double>::zeros(960, maxDCount);
	rightEyeDictionary = cv::Mat_<double>::zeros(960, maxDCount);
	leftGazeDictionary.resize(Eigen::NoChange, maxDCount);
	rightGazeDictionary.resize(Eigen::NoChange, maxDCount);

	loadConfigData(); // load calibration data

	dcount = 0; // counter to keep track of number of calibration images
	// cout << "finished eye init" << endl;
}

void eyeModel::loadConfigData() {
	ifstream* infile = new ifstream;
	string filename = "..\\..\\data\\calibration.txt";
	infile->open(filename);
	if (!infile->is_open()) {
		cout << "Could not open calibration file!" << endl;
	}
	string line;

	getline(*infile, line, ':');
	getline(*infile, line);
	calibParams.Tpix2cam = readMatrix(infile, 3, 3);

	getline(*infile, line, ':');
	getline(*infile, line);
	calibParams.Tcam2pix = readMatrix(infile, 3, 3);

	getline(*infile, line, ':');
	getline(*infile, line);
	calibParams.monitorPlane = readMatrix(infile, 1, 4).transpose();

	cout << "number of calibration points:" << endl << numOfCalibPoints << endl;
	cout << "samples per calibration point:" << endl << samplesPerCalibPoint << endl;
	//cout << "affineTrans:" << endl << calibParams.affineTrans << endl;
	//cout << "basis:" << endl << calibParams.basis << endl;
	//cout << "monitorPlane:" << endl << calibParams.monitorPlane << endl;
	//cout << "zBias:" << endl << calibParams.zbias << endl;

	infile->close();
	delete infile;
}

void eyeModel::createEyeModel(const Vertices& lm3D) {
	Eigen::MatrixXf leftEyeLm3D = lm3D.block<3, 6>(0, 26 - 1);
	Eigen::MatrixXf rightEyeLm3D = lm3D.block<3, 6>(0, 20 - 1);
	getEyeRegion(leftEyeLm3D, leftEyeRegion);
	getEyeRegion(rightEyeLm3D, rightEyeRegion);
}

void eyeModel::getEyeRegion(const Eigen::MatrixXf& eyeLm3D, eyeRegion& eyeReg) {
	float offset = 5.0f;
	float xmin = eyeLm3D.block<1, 6>(0, 0).minCoeff() - offset;
	float xmax = eyeLm3D.block<1, 6>(0, 0).maxCoeff() + offset;
	float ymin = eyeLm3D.block<1, 6>(1, 0).minCoeff();
	float ymax = eyeLm3D.block<1, 6>(1, 0).maxCoeff();

	float xd = xmax - xmin;
	float yd = ymax - ymin;
	float r = (float)eyeRegionRes.height / (float)eyeRegionRes.width;

	if (yd / xd > r) {
		xd = yd / r;
	}
	else {
		yd = r * xd;
	}

	eyeReg.center = eyeLm3D.rowwise().mean();
	eyeReg.width = xd;
	eyeReg.height = yd;
}

void eyeModel::calibrate(const Vertices& pc, const cv::Mat& color, const TransformType& trans, const cv::Point& pt) {
	if (dcount < maxDCount) { //numOfCalibPoints * samplesPerCalibPoint) {
		cv::Mat tex;
		computeTexture(color, tex);
		colorPointCloud cpc_raw(pc, tex);
		colorPointCloud cpc_clean;
		removeMissingData(cpc_raw, cpc_clean);
		
		Vertices pc_aligned;
		applyTransform(cpc_clean.pc, trans, pc_aligned);
		cpc_clean.pc = pc_aligned;

		leftEyeImg = createEyeImage(cpc_clean, leftEyeRegion);
		rightEyeImg = createEyeImage(cpc_clean, rightEyeRegion);

		cv::Mat xl, xr;
		leftEyeImg.reshape(0, 960).convertTo(xl, CV_64FC1);
		rightEyeImg.reshape(0, 960).convertTo(xr, CV_64FC1);

		xl = xl / norm(xl);
		xr = xr / norm(xr);
		
		xl.copyTo(leftEyeDictionary.col(dcount));
		xr.copyTo(rightEyeDictionary.col(dcount));

		// get inverse transform
		Vertices fp = pixelToCoord(pt);
		Vertices el, er;
		applyInvTransform(leftEyeRegion.center, trans, el);
		applyInvTransform(rightEyeRegion.center, trans, er);

		// store in dictionary
		Vertices lgv = trans.first.cast<float>() * computeNormalizedVector(el, fp);
		Vertices rgv = trans.first.cast<float>() * computeNormalizedVector(er, fp);
		//cout << "left gaze vector: " << lgv << endl;
		//cout << "right gaze vector: " << rgv << endl;

		leftGazeDictionary.col(dcount) = lgv;
		rightGazeDictionary.col(dcount) = rgv;
		dcount++;

	}
}

void eyeModel::run(const Vertices& pc, const cv::Mat& color, const TransformType& trans){
	cv::Mat tex;
	computeTexture(color, tex);
	colorPointCloud cpc_raw(pc, tex);
	colorPointCloud cpc_clean;
	removeMissingData(cpc_raw, cpc_clean);

	Vertices pc_aligned;
	applyTransform(cpc_clean.pc, trans, pc_aligned);
	cpc_clean.pc = pc_aligned;

	leftEyeImg = createEyeImage(cpc_clean, leftEyeRegion);
	rightEyeImg = createEyeImage(cpc_clean, rightEyeRegion);

	runOmp();
	leftFixPoint = computeFixPoint(leftGazeVector, leftEyeRegion.center, trans);
	rightFixPoint = computeFixPoint(rightGazeVector, rightEyeRegion.center, trans);
	//cout << "left fixation point: " << leftFixPoint << endl;
	//cout << "right fixation point: " << rightFixPoint << endl;
}

cv::Mat eyeModel::createEyeImage(const colorPointCloud& colorPc, const eyeRegion& eye) {
	// crop point cloud and texture to eye region
	colorPointCloud cpcCropped;
	cv::Rect r;
	cropEye(colorPc, eye, cpcCropped, r);

	// interpolate to create eye image
	return interpolateEyeImage(cpcCropped, eye, r);
}

void eyeModel::cropEye(const colorPointCloud& cpc, const eyeRegion& eye, colorPointCloud& cpcCropped, cv::Rect& rect) {

	// set boundaries
	float offset = 5.0f;
	float xmin = std::floorf(eye.center(0) - 0.5 * eye.width - offset);
	float xmax = std::ceilf(eye.center(0) + 0.5 * eye.width + offset);
	float ymin = std::floorf(eye.center(1) - 0.5 * eye.height - offset);
	float ymax = std::ceilf(eye.center(1) + 0.5 * eye.height + offset);

	rect = cv::Rect((int)xmin, (int)ymin, (int)(xmax - xmin), (int)(ymax - ymin));
	int n = cpc.pc.cols();
	//std::cout << "size: " << cpc.pc.rows() << " x " << cpc.pc.cols() << std::endl;
	std::vector<int> ind;
	// compute size of cropped region
	for (int i = 0; i < n; i++) {
		if (cpc.pc(0, i) > xmin && cpc.pc(0, i) < xmax && cpc.pc(1, i) > ymin && cpc.pc(1, i) < ymax) {
			ind.push_back(i);
		}
	}
	int nc = ind.size();
	//std::cout << "cropped size: " << nc << std::endl;
	// std::cout << "copying data ..." << std::endl;
	// copy the data 
	cpcCropped.pc.resize(3, nc);
	cpcCropped.tex = cv::Mat::zeros(1, nc, CV_8UC1);
	for (int i = 0; i < nc; i++) {
		cpcCropped.pc.col(i) = cpc.pc.col(ind.at(i));
		cpc.tex.col(ind.at(i)).copyTo(cpcCropped.tex.col(i));
	}
	// std::cout << "finished cropping eye ..." << std::endl;
}

cv::Mat eyeModel::interpolateEyeImage(const colorPointCloud& cpc, const eyeRegion& eye, const cv::Rect& r) {
	// Coordinates of data points.
	int n = cpc.pc.cols();
	std::vector<cv::Point2f> points;
	std::vector<uchar> val;

	for (int i = 0; i < n; i++) {
		points.push_back(cv::Point2f(cpc.pc(0, i), cpc.pc(1, i)));
		val.push_back(cpc.tex.at<uchar>(0, i));
	}

	// std::cout << "setting up algorithm ..." << std::endl;
	cv::Subdiv2D subdiv(r);
	subdiv.insert(points);

	int ni = eyeRegionRes.width * eyeRegionRes.height;
	//cv::Mat eyeImg = cv::Mat::zeros(1, ni, CV_8UC1);
	cv::Mat eyeImg = cv::Mat::zeros(eyeRegionRes.height, eyeRegionRes.width, CV_8UC1);
	Eigen::VectorXf x, y;
	x.setLinSpaced(eyeRegionRes.width, eye.center(0) - 0.5 * eye.width, eye.center(0) + 0.5 * eye.width);
	y.setLinSpaced(eyeRegionRes.height, eye.center(1) - 0.5 * eye.height, eye.center(1) + 0.5 * eye.height);
	// std::cout << "interpolation ..." << std::endl;
	uchar uc;
	// cout << x.size() << endl;
	for (int j = 0; j < y.size(); j++) {
		for (int i = 0; i < x.size(); i++) {
			// Get interpolated value at arbitrary location.
			cv::Point2f p(x(i), y(j));
			
			if (interpolateScatteredData(subdiv, p, val, uc)) {
				eyeImg.at<uchar>(y.size() - j - 1, i) = uc;
			}
		}
	}
	// std::cout << "finished interpolation ..." << std::endl;
	return eyeImg;
}

void eyeModel::getEyeImage(cv::Mat& img, int id) {
	if (id == 0) {
		img = leftEyeImg;
	}
	else if (id == 1) {
		img = rightEyeImg;
	}
	else if (id == 2) {
		img = leftEyeImgRec;
	}
	else if (id == 3) {
		img = rightEyeImgRec;
	}
}

void eyeModel::computeTexture(const cv::Mat& color, cv::Mat& tex) {
	cv::Mat gray;
	cv::cvtColor(color, gray, CV_RGB2GRAY);
	tex = gray.reshape(0, 1);
}

void eyeModel::removeMissingData(const colorPointCloud& in, colorPointCloud& out) {
	int n = in.pc.cols();
	int numMissing = 0;
	// check number of missing values
	for (int i = 0; i < n; i++) {
		if (in.pc(2, i) < 50.0f) {
			numMissing++;
		}
	}
	// std::cout << "missing data points: " << numMissing << std::endl;
	out.pc.resize(3, n - numMissing);
	out.tex = cv::Mat::zeros(1, n - numMissing, CV_8UC1);
	int count = 0;
	for (int i = 0; i < n; i++) {
		if (in.pc(2, i) > 50.0f) {
			out.pc.col(count) = in.pc.col(i);
			in.tex.col(i).copyTo(out.tex.col(count));
			count++;
		}
	}
}

void eyeModel::runOmp() {
	cv::Mat xl, xr; // imput image reshaped as vector
	cv::Mat_<double> cl, cr; // omp coefficients to be computed
	int k = 25; // omp sparsity parameter
	// reshape
	leftEyeImg.reshape(0, 960).convertTo(xl, CV_64FC1);
	rightEyeImg.reshape(0, 960).convertTo(xr, CV_64FC1);
	// normalize
	xl = xl / cv::norm(xl);
	xr = xr / cv::norm(xr);
	// run omp
	OMP(xl, leftEyeDictionary.colRange(0, dcount - 1), k, cl);
	OMP(xr, rightEyeDictionary.colRange(0, dcount - 1), k, cr);

	//std::cout << "left coeff: " << cl << std::endl;
	//std::cout << "right coeff: " << cr << std::endl;
	// get reconstructed image vector
	cv::Mat imgrec_l = leftEyeDictionary.colRange(0, dcount - 1) * cl;
	cv::Mat imgrec_r = rightEyeDictionary.colRange(0, dcount - 1) * cr;
	// get reconstructed gaze vector
	Eigen::Map<Eigen::VectorXd> cl_eigen(cl.ptr<double>(), cl.rows, cl.cols);
	Eigen::Map<Eigen::VectorXd> cr_eigen(cr.ptr<double>(), cr.rows, cr.cols);
	leftGazeVector = leftGazeDictionary.block(0, 0, 3, dcount) * cl_eigen.cast<float>();
	rightGazeVector = rightGazeDictionary.block(0, 0, 3, dcount) * cr_eigen.cast<float>();
	// reshape and normalize
	double min, max;
	cv::minMaxLoc(imgrec_l, &min, &max, NULL, NULL);
	cv::Mat irl = imgrec_l.reshape(0, 24) / max * 255;

	cv::minMaxLoc(imgrec_r, &min, &max, NULL, NULL);
	cv::Mat irr = imgrec_r.reshape(0, 24) / max * 255;

	//cout << irl << endl;
	//cout << irr << endl;
	irl.convertTo(leftEyeImgRec, CV_8UC1);
	irr.convertTo(rightEyeImgRec, CV_8UC1);

}

cv::Point eyeModel::computeFixPoint(const Vertices& gazeVec, const Vertices& eyeCenter, const TransformType& trans) {
	Vertices gazeVecAdjust, eyeCenterAdjust;
	applyInvTransform(gazeVec, std::make_pair(trans.first, Eigen::Vector3d::Zero()), gazeVecAdjust);
	applyInvTransform(eyeCenter, trans, eyeCenterAdjust);
	// compute intersection with monitor
	Vertices fixPoint = computePointPlaneIntersection(eyeCenterAdjust, gazeVecAdjust, calibParams.monitorPlane);
	// cout << "fix point: " << fixPoint << endl;
	return coordToPixel(fixPoint);
}

cv::Point eyeModel::coordToPixel(const Vertices& pt3) {
	Vertices pt2 = calibParams.Tcam2pix * pt3;
	return cv::Point(pt2(0), pt2(1));
}

Vertices eyeModel::pixelToCoord(const cv::Point& pt2) {
	Eigen::Vector3f tmp;
	tmp << pt2.x, pt2.y, 1;
	Vertices pt3 = calibParams.Tpix2cam * tmp;
	return pt3;
}

void eyeModel::getFixPoint(cv::Point& p, int id) {
	if (id == 0) {
		p = leftFixPoint;
	}
	else if (id == 1) {
		p = rightFixPoint;
	}
}