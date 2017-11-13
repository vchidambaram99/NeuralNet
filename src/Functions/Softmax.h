#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <cmath>
#include <unsupported/Eigen/CXX11/Tensor>
#include "ActivationFunction.h"

template<int inputDims>
class Softmax : public ActivationFunction<inputDims>{
    explicit Softmax(int a = 0){
        dimension = a;
    }
    Eigen::Tensor<double,inputDims> operator()(Eigen::Tensor<double,inputDims> input){
        //TODO
    }
    int dimension;
};

#endif //SOFTMAX_H

/* This was the old function, this will help when I write the above thing
template<int A>
Eigen::Tensor<double,A> softmax(Eigen::Tensor<double,A> input){
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
*/
