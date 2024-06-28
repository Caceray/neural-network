#include "algebra.hpp"
#include <cmath>
#include <Eigen/Dense>
#include <iostream>

#include <cassert>

using namespace std;

using Eigen::MatrixXf;
using Eigen::VectorXf;

void Sigmoid::main(VectorXf& input) const
{
    input = input.unaryExpr( [](float x){return 1 / (1+exp(-x));} );
}

void Sigmoid::prim(const VectorXf& activation, VectorXf& output) const
{
    output = activation.array() * (1-activation.array());
}

Activation* Sigmoid::clone() const
{
    return new Sigmoid();
}

void Softmax::main(VectorXf& input) const
{
    // Normalize before computing softmax
    input.array() -= input.maxCoeff();

    // Softmax formula
    input = input.unaryExpr( [](float x){return exp(x);} );
    input /= input.sum();
}

void Softmax::prim(const VectorXf& activation, VectorXf& output) const
{
    output = activation.array() * (1-activation.array());
}

Activation* Softmax::clone() const
{
    return new Softmax();
}

Quadratic::Quadratic(VectorXf* derivative):m_derivative(derivative){}

void Quadratic::getGradient(const VectorXf& computedOutput, const VectorXf& expectedOutput, VectorXf& result) const
{
    // NablaC = x-y
    result = (computedOutput-expectedOutput).array() * this->m_derivative->array();
}

CrossEntropy::CrossEntropy(){}

void CrossEntropy::getGradient(const VectorXf& computedOutput, const VectorXf& expectedOutput, VectorXf& result) const
{
    result = (computedOutput-expectedOutput).array();
}
