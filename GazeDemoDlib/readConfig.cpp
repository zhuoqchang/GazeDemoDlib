#include "readConfig.h"

using namespace std;
using namespace Eigen;

MatrixXf readMatrix(ifstream* f, int rows, int cols) {

	MatrixXf m(rows, cols);

	// Read numbers from file into buffer.
	for (int i = 0; i < rows; i++) {
		string line;
		getline(*f, line);
		stringstream stream(line);
		for (int j = 0; j < cols; j++) {
			stream >> m(i, j);
		}
	}

	return m;
};

float readNumber(ifstream* f) {
	float fl;
	string line;
	getline(*f, line);
	stringstream stream(line);
	stream >> fl;

	return fl;
};