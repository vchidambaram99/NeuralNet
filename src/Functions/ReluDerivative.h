#ifndef RELUDERIVATIVE_H
#define RELUDERIVATIVE_H

#include <unsupported/Eigen/CXX11/Tensor>
#include "ActivationFunction.h"

template<int inputDims>
class ReluDerivative : public ActivationFunction<inputDims>{
    Eigen::Tensor<double,inputDims> operator()(Eigen::Tensor<double,inputDims> input){
        double* d = input.data();
        for(int i = 0;i<input.size();i++){ //iterates over all elements
            d[i] = (d[i]>0)?1:0; //calculates relu
        }
    	return input;
    }
};

#endif //RELUDERIVATIVE_H
