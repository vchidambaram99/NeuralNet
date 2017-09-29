/*
 * BasicNeuralNet.cpp
 *
 *  Created on: Sep 28, 2017
 *      Author: vchid
 */

#include "BasicNeuralnet.hpp"
#include <random>
#include <ctime>

static void randomInitialize(Eigen::MatrixXd& matrix, double mean, double stdDev, double shift){
	std::default_random_engine generator(srand((unsigned)time(NULL)));
	std::normal_distribution dist(mean,stdDev);
	for(int i = 0;i<matrix.rows();i++){
		for(int j = 0;j<matrix.cols();j++){
			matrix(i,j) = dist(generator)+shift;
		}
	}
}
static void randomInitialize(Eigen::VectorXd& matrix, double mean, double stdDev, double shift){
	std::default_random_engine generator(srand((unsigned)time(NULL)));
	std::normal_distribution dist(mean,stdDev);
	for(int i = 0;i<matrix.rows();i++){
		for(int j = 0;j<matrix.cols();j++){
			matrix(i,j) = dist(generator)+shift;
		}
	}
}

BasicNeuralNet::BasicNeuralNet(){
	costFunction = nullptr;
	bias = true;
}
BasicNeuralNet::BasicNeuralNet(std::vector<int> layerSizes, std::vector<Eigen::MatrixXd (*)(Eigen::MatrixXd)> _activationFunctions,
				   std::vector<Eigen::MatrixXd (*)(Eigen::MatrixXd)> _derivatives,
				   Eigen::MatrixXd (*_costFunction) (Eigen::MatrixXd,Eigen::MatrixXd), bool _bias){
	bias = _bias;
	costFunction = _costFunction;
	weights = std::vector(layerSizes.size()-1);
	std::vector<Eigen::MatrixXd (*)(Eigen::MatrixXd)> activationFunctions = _activationFunctions;
	std::vector<Eigen::MatrixXd (*)(Eigen::MatrixXd)> derivatives = _derivatives;
	for(int i = 0;i<layerSizes.size()-1;i++){
		weights[i] = Eigen::MatrixXd(layerSizes[i],layerSizes[i+1]);
		randomInitialize(weights[i],0,1,0);
	}
	if(bias){
		biases = std::vector(layerSizes.size()-1);
		for(int i = 0;i<layerSizes.size()-1;i++){
			biases[i] = Eigen::VectorXd(layerSizes[i+1]);
			randomInitialize(biases[i],0,1,0);
		}
	}
}
Eigen::VectorXd BasicNeuralNet::fire(Eigen::MatrixXd input){ //columns of input are each case (can do multiple cases simultaneously)
	for(int i = 0;i<weights.size();i++){
		if(bias){
			input = (*activationFunctions[i])((weights[i]*input).colwise()+biases[i]);
		}else{
			input = (*activationFunctions[i])(weights[i]*input);
		}
	}
	return input;
}
Eigen::MatrixXd BasicNeuralNet::backprop(Eigen::MatrixXd input, Eigen::MatrixXd answer){
	std::vector<Eigen::MatrixXd> activations = std::vector<Eigen::MatrixXd>(weights.size()+1); //activations of all (even input) layers
	std::vector<Eigen::MatrixXd> derivatives = std::vector<Eigen::MatrixXd>(weights.size());   //derivatives of all non-input layers
	std::vector<Eigen::MatrixXd> errors 	 = std::vector<Eigen::MatrixXd>(weights.size());   //errors of all non-input layers
	activations[0] = &input;
	for(int i = 0;i<weights.size();i++){
		Eigen::MatrixXd weighted;
		if(bias)weighted = (weights[i]*activations[i]).colwise()+biases[i];
		else weighted = weights[i]*activations[i];
		activations[i+1] = (*activationFunctions[i])(weighted);
		derivatives[i] = (*derivatives[i])(weighted);
	}
	errors.back() = costFunction(activations.back(),answer).cwiseProduct(derivatives.back()); //calculates error of last layer
	for(int i = errors.size()-2;i>=0;i--){
		errors[i] = (weights[i+1].transpose()*errors[i+1]).cwiseProduct(derivatives[i]); //calculates error of subsequent layers
	}
	if(bias){
		for(int i = 0;i<biases.size();i++){
			biases[i]+=errors[i].rowwise().sum(); //updates biases
		}
	}
	for(int i = 0;i<weights.size();i++){
		weights[i]+=errors[i]*activations[i].transpose(); //updates weights
	}
	return activations.back();
}


