#include "algebra.hpp"
#include <cmath>
#include <Eigen/Dense>
#include <iostream>

#include <cassert>

using namespace std;

using Eigen::MatrixXf;
using Eigen::VectorXf;

void Sigmoid::main(VectorXf& input, VectorXf& output) const
{
    output = input.unaryExpr( [](float x){return 1 / (1+exp(-x));} );
}

void Sigmoid::prim(VectorXf& input, VectorXf& output) const
{
    VectorXf S; this->main(input, S);
    output = S.array() * (1-S.array());
}

void Softmax::main(VectorXf& input, VectorXf& output) const
{
    // Normalize before computing softmax
    input.array() -= input.maxCoeff();

    // Softmax formula
    output = input.unaryExpr( [](float x){return exp(x);} );
    output /= output.sum();
}

void Softmax::prim(VectorXf& input, VectorXf& output) const
{
    VectorXf S; this->main(input, S);
    output = S.array() * (1-S.array());
}

Quadratic::Quadratic(VectorXf* derivative):m_derivative(derivative){}

void Quadratic::gradient(const VectorXf &x, const VectorXf &y, VectorXf* result) const
{
    // NablaC = x-y
    *result = (x-y).array() * this->m_derivative->array();
}

CrossEntropy::CrossEntropy(){}

void CrossEntropy::gradient(const VectorXf &x, const VectorXf &y, VectorXf* result) const
{
    *result = (x-y).array() / 2;
}
