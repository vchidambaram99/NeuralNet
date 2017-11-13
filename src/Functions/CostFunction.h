#ifndef COSTFUNCTION_H
#define COSTFUNCTION_H

#include <unsupported/Eigen/CXX11/Tensor>

template<int A>
class CostFunction{
    virtual Eigen::Tensor<double,A> operator()(Eigen::Tensor<double,A> input, const Eigen::Tensor<double,A>& answer) = 0;
};

#endif //COSTFUNCTION_H
