/*
 * BasicNeuralNet.cpp
 *
 *  Created on: Sep 28, 2017
 *      Author: vchid
 */

#include "BasicNeuralNet.h"
#include <iostream>
#include <random>
#include <ctime>

static void randomInitialize(Eigen::MatrixXd& matrix, double mean, double stdDev, double shift){//initializes values of matrix randomly on a normal distribution
	std::default_random_engine generator((unsigned)time(0));
	std::normal_distribution<double> dist(mean,stdDev);
	for(int i = 0;i<matrix.rows();i++){
		for(int j = 0;j<matrix.cols();j++){
			matrix(i,j) = dist(generator)+shift;
		}
	}
}
static void randomInitialize(Eigen::VectorXd& matrix, double mean, double stdDev, double shift){//same as above for vectors
	std::default_random_engine generator((unsigned)time(0));
	std::normal_distribution<double> dist(mean,stdDev);
	for(int i = 0;i<matrix.rows();i++){
		for(int j = 0;j<matrix.cols();j++){
			matrix(i,j) = dist(generator)+shift;
		}
	}
}

BasicNeuralNet::BasicNeuralNet(){//empty constructor
	cost = nullptr;
	bias = true;
	learnRate = 0.05;
}
BasicNeuralNet::BasicNeuralNet(std::vector<int> layerSizes, std::vector<activationFunction> _activationFunctions,
							   std::vector<activationFunction> _derivativeFunctions, double _learnRate,
							   Eigen::MatrixXd (*_costFunction) (Eigen::MatrixXd,Eigen::MatrixXd&), bool _bias){
	bias = _bias;
	learnRate = _learnRate;
	cost = _costFunction;
	weights = std::vector<Eigen::MatrixXd>(layerSizes.size()-1);
	activationFunctions = _activationFunctions;
	derivativeFunctions = _derivativeFunctions;
	for(int i = 0;i<layerSizes.size()-1;i++){
		weights[i] = Eigen::MatrixXd(layerSizes[i+1],layerSizes[i]);
		randomInitialize(weights[i],0,1,0);
	}
	if(bias){
		biases = std::vector<Eigen::VectorXd>(layerSizes.size()-1);
		for(int i = 0;i<layerSizes.size()-1;i++){
			biases[i] = Eigen::VectorXd(layerSizes[i+1]);
			randomInitialize(biases[i],0,1,0);
		}
	}
}
Eigen::MatrixXd BasicNeuralNet::fire(Eigen::MatrixXd input){ //columns of input are each case (can do multiple cases simultaneously or SGD)
	for(int i = 0;i<weights.size();i++){//runs the activation function on the weighted matrices and iterates through network
		if(bias){
			input = activationFunctions[i]((weights[i]*input).colwise()+biases[i]);
		}else{
			input = activationFunctions[i](weights[i]*input);
		}
	}
	return input;
}
Eigen::MatrixXd BasicNeuralNet::backprop(Eigen::MatrixXd input, Eigen::MatrixXd answer){//updates weights and biases given answers and input
	std::vector<Eigen::MatrixXd> activations = std::vector<Eigen::MatrixXd>(weights.size()+1); //activations of all (even input) layers
	std::vector<Eigen::MatrixXd> derivatives = std::vector<Eigen::MatrixXd>(weights.size());   //derivatives of all non-input layers
	std::vector<Eigen::MatrixXd> errors 	 = std::vector<Eigen::MatrixXd>(weights.size());   //errors of all non-input layers
	activations[0] = input;
	for(int i = 0;i<weights.size();i++){
		Eigen::MatrixXd weighted;
		if(bias)weighted = (weights[i]*activations[i]).colwise()+biases[i];
		else weighted = weights[i]*activations[i];
		activations[i+1] = activationFunctions[i](weighted);//saves the activation of this layer
		derivatives[i] = derivativeFunctions[i](weighted);//saves the derivatives of this layer
	}
	errors.back() = (cost(activations.back(),answer)).cwiseProduct(derivatives.back()); //calculates error of last layer
	for(int i = errors.size()-2;i>=0;i--){
		errors[i] = (weights[i+1].transpose()*errors[i+1]).cwiseProduct(derivatives[i]); //calculates error of subsequent layers
	}
	if(bias){
		for(int i = 0;i<biases.size();i++){
			biases[i]-=errors[i].rowwise().sum()*learnRate; //updates biases
		}
	}
	for(int i = 0;i<weights.size();i++){
		weights[i]-=errors[i]*activations[i].transpose()*learnRate; //updates weights
	}
	return activations.back();//returns the output of the net (what fire() would return), so it can be used if necessary
}
