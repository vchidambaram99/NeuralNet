#ifndef DENSE_H
#define DENSE_H
#include <Layer.h>

template<int inputDims>
class Dense : public Layer<inputDims>{
    
    Eigen::Tensor<double, inputDims> fire(Eigen::Tensor<double,inputDims> input){

    }
    Eigen::Tensor<double, inputDims> derivative(Eigen::Tensor<double,inputDims> input){

    }
};

#endif //DENSE_H
