#ifndef SOFTMAXDERIVATIVE_H
#define SOFTMAXDERIVATIVE_H

#include <cmath>
#include <unsupported/Eigen/CXX11/Tensor>
#include "ActivationFunction.h"

template<int inputDims>
class SoftmaxDerivative : public ActivationFunction<inputDims>{
    explicit SoftmaxDerivative(int a = 0){
        dimension = a;
    }
    Eigen::Tensor<double,inputDims> operator()(Eigen::Tensor<double,inputDims> input){
        //TODO
    }
    int dimension;
};

#endif //SOFTMAXDERIVATIVE_H

/* this is the old version and will help when I write the above thing
template<int A>
Eigen::Tensor<double,A> softmaxDerivative(Eigen::Tensor<double,A> input){
    for(int i = 0;i<input.cols();i++){
		double expSum = 0;
		for(int j = 0;j<input.rows();j++){
			expSum+=exp(input(j,i));
		}
		for(int j = 0;j<input.rows();j++){
			double a = exp(input(i,j))/expSum;
			input(i,j) = a-(a*a);
		}
	}
	return input;
}
*/
