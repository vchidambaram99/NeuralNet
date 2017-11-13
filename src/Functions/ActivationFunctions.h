/*
 * ActivationFunctions.h
 *
 *  Created on: Sep 28, 2017
 *      Author: vchid
 */

#ifndef NEURALNETS_ACTIVATIONFUNCTIONS_H_
#define NEURALNETS_ACTIVATIONFUNCTIONS_H_
#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>

template<int A>
Eigen::Tensor<double,A> sigmoid(Eigen::Tensor<double,A> input){//applies sigmoid to every element
    double* d = input.data();
    for(int i = 0;i<input.size();i++){ //iterates over all elements
        d[i] = 1/(1+exp(-d[i])); //calculates sigmoid
    }
	return input;
}

template<int A>
Eigen::Tensor<double,A> sigmoidDerivative(Eigen::Tensor<double,A> input){//applies sigmoid derivative to every element
    double* d = input.data();
    for(int i = 0;i<input.size();i++){ //iterates over all elements
        double temp = 1/(1+exp(-d[i])); //calculates sigmoid derivative
        d[i] = temp*(1-temp);
    }
	return input;
}

template<int A>
Eigen::Tensor<double,A> relu(Eigen::Tensor<double,A> input){//applies relu to every element
    double* d = input.data();
    for(int i = 0;i<input.size();i++){ //iterates over all elements
        d[i] = (d[i]>0)?d[i]:0; //calculates relu
    }
	return input;
}

template<int A>
Eigen::Tensor<double,A> reluDerivative(Eigen::Tensor<double,A> input){//applies relu derivative to every element
    double* d = input.data();
    for(int i = 0;i<input.size();i++){ //iterates over all elements
        d[i] = (d[i]>0)?1:0; //calculates relu derivative
    }
	return input;
}

template<int A>
Eigen::Tensor<double,A> softmax(Eigen::Tensor<double,A> input){//TODO fix
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

template<int A>
Eigen::Tensor<double,A> softmaxDerivative(Eigen::Tensor<double,A> input){//TODO fix
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

template<int A>
Eigen::Tensor<double,A>(*activationFunction(std::string s))(Eigen::MatrixXd){//returns function pointer to appropriate activation function given name
	if(s=="sigmoid"){
        return sigmoid<A>;
    }else if(s=="relu"){
        return relu<A>;
    }else if(s=="softmax"){
        return softmax<A>;
    }
	return 0;
}

template<int A>
Eigen::Tensor<double,A>(*activationDerivative(std::string s))(Eigen::MatrixXd){//returns function pointer to appropriate activation derivative given name
    if(s=="sigmoid"){
        return sigmoidDerivative;
    }else if(s=="relu"){
        return reluDerivative;
    }else if(s=="softmax"){
        return softmaxDerivative;
    }
	return 0;
}

#endif /* NEURALNETS_ACTIVATIONFUNCTIONS_H_ */
