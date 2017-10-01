/*
 * ActivationFunctions.hpp
 *
 *  Created on: Sep 28, 2017
 *      Author: vchid
 */

#ifndef NEURALNETS_ACTIVATIONFUNCTIONS_HPP_
#define NEURALNETS_ACTIVATIONFUNCTIONS_HPP_
#include <Eigen/Dense>
#include <cmath>

Eigen::MatrixXd sigmoid(Eigen::MatrixXd input);
Eigen::MatrixXd sigmoidDerivative(Eigen::MatrixXd input);
Eigen::MatrixXd relu(Eigen::MatrixXd input);
Eigen::MatrixXd reluDerivative(Eigen::MatrixXd input);
Eigen::MatrixXd softmax(Eigen::MatrixXd input);
Eigen::MatrixXd softmaxDerivative(Eigen::MatrixXd input);

const int sigmoidNum = 0;
const int reluNum = 1;
const int softmaxNum = 2;

Eigen::MatrixXd(*activationFunctionByNum(int a))(Eigen::MatrixXd);
Eigen::MatrixXd(*activationDerivativeByNum(int a))(Eigen::MatrixXd);

#endif /* NEURALNETS_ACTIVATIONFUNCTIONS_HPP_ */
