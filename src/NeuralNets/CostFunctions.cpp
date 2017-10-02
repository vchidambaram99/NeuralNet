/*
 * CostFunctions.cpp
 *
 *  Created on: Sep 28, 2017
 *      Author: vchid
 */
#include "CostFunctions.hpp"

Eigen::MatrixXd quadraticCostDerivative(Eigen::MatrixXd input, Eigen::MatrixXd &answer){
	return input-answer;
}
Eigen::MatrixXd crossEntropyCostDerivative(Eigen::MatrixXd input, Eigen::MatrixXd &answer){
	//matrix form is slower unless going directly to changes for weights (which wouldn't work for how I set up the BasicNeuralNet class)
	//conveniently, this also happens to work for the negative log likelihood cost function for softmax layers (because calculus)
	for(int i = 0;i<input.rows();i++){
		for(int j = 0;j<input.cols();j++){
			input(i,j) = (input(i,j)-answer(i,j))/(input(i,j)*(1-input(i,j)));
		}
	}
	return input;
}

Eigen::MatrixXd(*costDerivativeByNum(int a))(Eigen::MatrixXd,Eigen::MatrixXd&){//returns appropriate cost function pointer give num
	switch(a){
	case quadCostNum:
		return quadraticCostDerivative;
	case crossEntropyNum:
		return crossEntropyCostDerivative;
	}
	return 0;
}
