#ifndef LAYER_H
#define LAYER_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <Functions/ActivationFunctions.h>

class BaseLayer {
public:
    virtual ~BaseLayer() {};
};

template<int inputDims>
class Layer : public BaseLayer{
public:
    virtual ~Layer();
    virtual Eigen::Tensor<double,inputDims> fire(Eigen::Tensor<double,inputDims>) = 0;
    virtual Eigen::Tensor<double,inputDims> derivative(Eigen::Tensor<double,inputDims>) = 0;
};

#endif //LAYER_H
