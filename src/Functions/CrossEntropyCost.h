#ifndef CROSSENTRYOPYCOST_H
#define CROSSENTRYOPYCOST_H

#include <unsupported/Eigen/CXX11/Tensor>
#include "CostFunction.h"

template<int A>
class CrossEntropyCost : public CostFunction<A>{ //Note: this is the derivative, not the actual cost
    Eigen::Tensor<double,A> operator()(Eigen::Tensor<double,A> input, const Eigen::Tensor<double,A>& answer){
        //matrix form is slower unless going directly to changes for weights (which wouldn't work for how I set up the BasicNeuralNet class)
    	//conveniently, this also happens to work for the negative log likelihood cost function for softmax layers (because calculus)
    	for(int i = 0;i<input.rows();i++){
    		for(int j = 0;j<input.cols();j++){
    			input(i,j) = (input(i,j)-answer(i,j))/(input(i,j)*(1-input(i,j)));
    		}
    	}
    	return input;
    }
};

#endif //CROSSENTRYOPYCOST_H
