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
#include <ctime>
#include <cstdlib>
using namespace std;

int main(){
	vector<Eigen::MatrixXd> trainLabels = readLabels("C:/Users/vchid/Downloads/c++net MNIST data/train-labels.idx1-ubyte");
	vector<Eigen::MatrixXd> testLabels  = readLabels("C:/Users/vchid/Downloads/c++net MNIST data/t10k-labels.idx1-ubyte");
	vector<Eigen::MatrixXd> trainImages = readImages("C:/Users/vchid/Downloads/c++net MNIST data/train-images.idx3-ubyte");
	vector<Eigen::MatrixXd> testImages  = readImages("C:/Users/vchid/Downloads/c++net MNIST data/t10k-images.idx3-ubyte");
	vector<int> ls = {784,30,10};
	vector<activationFunction> at = {sigmoid,sigmoid};
	vector<activationFunction> dt = {sigmoidDerivative,sigmoidDerivative};
	BasicNeuralNet bnn(ls,at,dt,.3,quadraticCostDerivative,true);
	int epochs = 30;
	int minibatchSize = 10;
	for(int i = 0;i<epochs;i++){
		srand(time(0));
		for(int j = trainLabels.size();j>1;j--){//Fisher-Yates shuffle of training data
			int index = rand()%j;
			Eigen::MatrixXd temp1 = trainLabels[index];
			trainLabels[index] = trainLabels[j-1];
			trainLabels[j-1] = temp1;
			Eigen::MatrixXd temp2 = trainImages[index];
			trainImages[index] = trainImages[j-1];
			trainImages[j-1] = temp2;
		}
		for(int j = 0;j<trainLabels.size()/minibatchSize;j++){
			Eigen::MatrixXd images(784,minibatchSize);//where minibatch images are stored
			Eigen::MatrixXd labels(10,minibatchSize);//where minibatch labels are stored
			for(int k = 0;k<minibatchSize;k++){
				images.col(k) = trainImages[(j*minibatchSize)+k];//adds each image
				labels.col(k) = trainLabels[(j*minibatchSize)+k];//adds each label
			}
			bnn.backprop(images,labels);//adjusts the weights for this minibatch
		}
		int correct = 0;
		for(int j = 0;j<testLabels.size();j++){//iterates over test cases
			Eigen::MatrixXd a = bnn.fire(testImages[j]);
			int maxindex = 0;
			for(int k = 1;k<10;k++){
				if(a(k)>a(maxindex)){
					maxindex = k;//the index of the max of the vector returned by the neural net is its answer
				}
			}
			if(testLabels[j](maxindex)==1){//counts number correct
				correct++;
			}
		}
		cout<<"Test accuracy (after epoch "<<i<<"): "<<correct<<"/10000."<<endl;;
	}
	return 0;
}


