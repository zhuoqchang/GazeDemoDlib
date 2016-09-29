#pragma once

#ifndef READCONFIG_H
#define READCONFIG_H

#include <fstream>
#include <string>
#include <Eigen/Dense>

Eigen::MatrixXf readMatrix(std::ifstream* f, int rows, int cols);

float readNumber(std::ifstream* f);

#endif