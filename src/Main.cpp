/*
 * Main.cpp
 *
 *  Created on: Sep 28, 2017
 *      Author: vchid
 */
#include "MNIST/MNISTReader.hpp"
#include "NeuralNets/BasicNeuralNet.hpp"
#include "NeuralNets/ActivationFunctions.hpp"
#include "NeuralNets/CostFunctions.hpp"
using namespace std;

int main(){
	Eigen::MatrixXd a(4,2);
	a<<1,2,3,4,5,6,7,8;
	vector<int> ls = {4,5,4};
	std::vector<activationFunction> af = {sigmoid,sigmoid};
	std::vector<activationFunction> df = {sigmoidDerivative,sigmoidDerivative};
	BasicNeuralNet bnn(ls,af,df,quadraticCostDerivative,true);
	cout<<bnn.fire(a)<<endl;
	return 0;
}


