#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H

#include <unsupported/Eigen/CXX11/Tensor>

template<int inputDims>
class ActivationFunction{
    virtual Eigen::Tensor<double,inputDims> operator()(Eigen::Tensor<double,inputDims> input) = 0;
};

#endif //ACTIVATIONFUNCTION_H
