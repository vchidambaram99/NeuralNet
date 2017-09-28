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

class BasicNeuralNet{
public:
	std::vector<Eigen::MatrixXd> weights;
	std::vector<Eigen::VectorXd> biases;
	std::vector<Eigen::MatrixXd (*)(Eigen::MatrixXd)> activationFunctions;
	std::vector<Eigen::MatrixXd (*)(Eigen::MatrixXd)> derivatives;
	bool bias;
	BasicNeuralNet(){
		bias = true;
	}
	BasicNeuralNet(std::vector<int> layerSizes,std::vector<Eigen::MatrixXd (*)(Eigen::MatrixXd)> _activationFunctions, std::vector<Eigen::MatrixXd (*)(Eigen::MatrixXd)> _derivatives ,bool _bias){
		//layerSizes is a vector of the size of each layer
		//activationFunctions is a vector to function pointers that apply to MatrixXds
		//derivatives is the derivatives of activationFunctions
		bias = _bias;
		weights = std::vector(layerSizes.size()-1);
		std::vector<Eigen::MatrixXd (*)(Eigen::MatrixXd)> activationFunctions = _activationFunctions;
		std::vector<Eigen::MatrixXd (*)(Eigen::MatrixXd)> derivatives = _derivatives;
		for(int i = 0;i<layerSizes.size()-1;i++){
			weights[i] = Eigen::MatrixXd(layerSizes[i],layerSizes[i+1]);
		}
		if(bias){
			biases = std::vector(layerSizes.size()-1);
			for(int i = 0;i<layerSizes.size()-1;i++){
				biases[i] = Eigen::VectorXd(layerSizes[i+1]);
			}
		}
	}
	Eigen::VectorXd fire(Eigen::VectorXd input){
		for(int i = 0;i<weights.size();i++){
			if(bias){
				input = (*activationFunctions[i])((weights[i]*input)+biases[i]);
			}else{
				input = (*activationFunctions[i])(weights[i]*input);
			}
		}
		return input;
	}
	Eigen::VectorXd backprop(){

	}
};

#endif /* NEURALNETS_BASICNEURALNET_HPP_ */
