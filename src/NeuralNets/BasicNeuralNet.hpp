/*
 * BasicNeuralNet.hpp
 *
 *  Created on: Sep 28, 2017
 *      Author: vchid
 */

#ifndef NEURALNETS_BASICNEURALNET_HPP_
#define NEURALNETS_BASICNEURALNET_HPP_
#include <vector>
#include <Eigen/Dense>

typedef Eigen::MatrixXd (*activationFunction)(Eigen::MatrixXd); //activation function pointers and their derivatives
typedef Eigen::MatrixXd (*costFunction)(Eigen::MatrixXd,Eigen::MatrixXd&); //cost function derivative pointers

class BasicNeuralNet{
public:
	std::vector<Eigen::MatrixXd> weights;
	std::vector<Eigen::VectorXd> biases;
	std::vector<activationFunction> activationFunctions;
	std::vector<activationFunction> derivativeFunctions;
	costFunction cost;
	bool bias;

	BasicNeuralNet();

	//layerSizes is a vector of the size of each layer
	//activationFunctions is a vector to function pointers that apply to MatrixXds
	//derivatives is the derivatives of activationFunctions
	BasicNeuralNet(std::vector<int> layerSizes, std::vector<activationFunction> _activationFunctions,
				   std::vector<activationFunction> _derivativeFunctions,
				   Eigen::MatrixXd (*_costFunction) (Eigen::MatrixXd,Eigen::MatrixXd&), bool _bias);

	Eigen::MatrixXd fire(Eigen::MatrixXd input);//columns of input are each case (can do multiple cases simultaneously)
	Eigen::MatrixXd backprop(Eigen::MatrixXd input, Eigen::MatrixXd answer);
};

#endif /* NEURALNETS_BASICNEURALNET_HPP_ */
