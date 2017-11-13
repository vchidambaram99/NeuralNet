#ifndef QUADRATICCOST_H
#define QUADRATICCOST_H

#include <unsupported/Eigen/CXX11/Tensor>
#include "CostFunction.h"

template<int A>
class QuadraticCost : public CostFunction<A>{ //Note: this is the derivative, not the actual cost
    Eigen::Tensor<double,A> operator()(Eigen::Tensor<double,A> input, const Eigen::Tensor<double,A>& answer){
        return input-answer;
    }
};

#endif //QUADRATICCOST_H
