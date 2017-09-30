/*
 * ActivationFunctions.cpp
 *
 *  Created on: Sep 28, 2017
 *      Author: vchid
 */
#include "ActivationFunctions.hpp"

Eigen::MatrixXd sigmoid(Eigen::MatrixXd input){
	for(int i = 0;i<input.rows();i++){
		for(int j = 0;j<input.cols();j++){
			input(i,j) = 1/(1+exp(-input(i,j)));
		}
	}
	return input;
}
Eigen::MatrixXd sigmoidDerivative(Eigen::MatrixXd input){
	for(int i = 0;i<input.rows();i++){
		for(int j = 0;j<input.cols();j++){
			double a = 1/(1+exp(-input(i,j)));
			input(i,j) = a*(1-a);
		}
	}
	return input;
}
Eigen::MatrixXd relu(Eigen::MatrixXd input){
	for(int i = 0;i<input.rows();i++){
		for(int j = 0;j<input.cols();j++){
			input(i,j) = (input(i,j)>0)?input(i,j):0;
		}
	}
	return input;
}
Eigen::MatrixXd reluDerivative(Eigen::MatrixXd input){
	for(int i = 0;i<input.rows();i++){
		for(int j = 0;j<input.cols();j++){
			input(i,j) = (input(i,j)>0)?1:0;
		}
	}
	return input;
}
Eigen::MatrixXd softmax(Eigen::MatrixXd input){
	for(int i = 0;i<input.cols();i++){
		double expSum = 0;
		for(int j = 0;j<input.rows();j++){
			expSum+=exp(input(j,i));
		}
		for(int j = 0;j<input.rows();j++){
			input(i,j) = exp(input(i,j))/expSum;
		}
	}
	return input;
}
Eigen::MatrixXd softmaxDerivative(Eigen::MatrixXd input){
	for(int i = 0;i<input.cols();i++){
		double expSum = 0;
		for(int j = 0;j<input.rows();j++){
			expSum+=exp(input(j,i));
		}
		for(int j = 0;j<input.rows();j++){
			double a = exp(input(i,j))/expSum;
			input(i,j) = a-(a*a);
		}
	}
	return input;
}
