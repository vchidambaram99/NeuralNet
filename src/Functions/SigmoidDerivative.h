#ifndef SIGMOIDDERIVATIVE_H
#define SIGMOIDDERIVATIVE_H

#include <cmath>
#include <unsupported/Eigen/CXX11/Tensor>
#include "ActivationFunction.h"

template<int inputDims>
class SigmoidDerivative : public ActivationFunction<inputDims>{
    Eigen::Tensor<double,inputDims> operator()(Eigen::Tensor<double,inputDims> input){
        double* d = input.data();
        for(int i = 0;i<input.size();i++){ //iterates over all elements
            double temp = 1/(1+exp(-d[i])); //calculates sigmoid derivative
            d[i] = temp*(1-temp);
        }
    	return input;
    }
};

#endif //SIGMOIDDERIVATIVE_H
