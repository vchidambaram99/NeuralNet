#ifndef DENSE_H
#define DENSE_H
#include <Layer.h>
#include <Functions/ActivationFunctions.h>

template<int inputDims>
class Dense : public Layer<inputDims>{
public:
    Dense(actFunc<double,inputDims> a,actFunc<double,inputDims> d){
        act = a;
        deriv = d;
    }
    ~Dense(){}
    Eigen::Tensor<double, inputDims> fire(Eigen::Tensor<double,inputDims> input){
        return activation(input);
    }
    Eigen::Tensor<double, inputDims> derivative(Eigen::Tensor<double,inputDims> input){
        return derivative(input);
    }
    actFunc<double,inputDims> act;
    actFunc<double,inputDims> deriv;
};

#endif //DENSE_H
