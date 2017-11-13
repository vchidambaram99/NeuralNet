#ifndef SIGMOID_H
#define SIGMOID_H

#include <cmath>
#include <unsupported/Eigen/CXX11/Tensor>
#include "ActivationFunction.h"

template<int inputDims>
class Sigmoid : public ActivationFunction<inputDims>{
    Eigen::Tensor<double,inputDims> operator()(Eigen::Tensor<double,inputDims> input){
        double* d = input.data();
        for(int i = 0;i<input.size();i++){ //iterates over all elements
            d[i] = 1/(1+exp(-d[i])); //calculates sigmoid
        }
    	return input;
    }
};

#endif //SIGMOID_H
