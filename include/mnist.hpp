#ifndef mnist_hpp
#define mnist_hpp

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include "dataset.hpp"

class MNIST
{
public:
    static void load(Dataset& dataset);
};
#endif /* mnist_hpp */
