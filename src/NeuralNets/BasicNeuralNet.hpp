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
	std::vector<Eigen::MatrixXd> biases;
	std::vector<Eigen::MatrixXd (*)(Eigen::MatrixXd)> activationFunctions;
	std::vector<Eigen::MatrixXd (*)(Eigen::MatrixXd)> derivatives;
	Eigen::MatrixXd (*costFunction)(Eigen::MatrixXd);
	bool bias;
	BasicNeuralNet(){
		costFunction = nullptr;
		bias = true;
	}
	BasicNeuralNet(std::vector<int> layerSizes, std::vector<Eigen::MatrixXd (*)(Eigen::MatrixXd)> _activationFunctions,
				   std::vector<Eigen::MatrixXd (*)(Eigen::MatrixXd)> _derivatives,
				   Eigen::MatrixXd (*_costFunction) (Eigen::MatrixXd,Eigen::MatrixXd), bool _bias){
		//layerSizes is a vector of the size of each layer
		//activationFunctions is a vector to function pointers that apply to MatrixXds
		//derivatives is the derivatives of activationFunctions
		bias = _bias;
		costFunction = _costFunction;
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
	Eigen::VectorXd fire(Eigen::MatrixXd input){//columns of input are each case (can do multiple cases simultaneously)
		for(int i = 0;i<weights.size();i++){
			if(bias){
				input = (*activationFunctions[i])((weights[i]*input)+biases[i]);
			}else{
				input = (*activationFunctions[i])(weights[i]*input);
			}
		}
		return input;
	}
	Eigen::MatrixXd backprop(Eigen::MatrixXd input, Eigen::MatrixXd answer){
		std::vector<Eigen::MatrixXd> activations = std::vector<Eigen::MatrixXd>(weights.size()+1);//activations of all (even input) layers
		std::vector<Eigen::MatrixXd> derivatives = std::vector<Eigen::MatrixXd>(weights.size());  //derivatives of all non-input layers
		std::vector<Eigen::MatrixXd> errors 	 = std::vector<Eigen::MatrixXd>(weights.size());  //errors of all non-input layers
		activations[0] = &input;
		for(int i = 0;i<weights.size();i++){
			Eigen::MatrixXd weighted;
			if(bias)weighted = (weights[i]*activations[i])+biases[i];
			else weighted = weights[i]*activations[i];
			activations[i+1] = (*activationFunctions[i])(weighted);
			derivatives[i] = (*derivatives[i])(weighted);
		}
		errors.back() = costFunction(activations.back(),answer).cwiseProduct(derivatives.back()); //calculates error of last layer
		for(int i = errors.size()-2;i>=0;i--){
			errors[i] = (weights[i+1].transpose()*errors[i+1]).cwiseProduct(derivatives[i]);
		}
		if(bias){
			for(int i = 0;i<biases.size();i++){
				biases[i]+=errors[i];
			}
		}
		for(int i = 0;i<weights.size();i++){
			weights[i]+=errors[i]*activations[i].transpose();
		}
		return activations.back();
	}
};

#endif /* NEURALNETS_BASICNEURALNET_HPP_ */
