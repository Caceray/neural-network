#ifndef algebra_hpp
#define algebra_hpp

#include <stdio.h>
#include <Eigen/Dense>

using Eigen::MatrixXf;
using Eigen::VectorXf;

enum class CostType : unsigned char
{
    Quadratic,
    CrossEntropy
};

enum class ActivationType : unsigned char
{
    Sigmoid,
    Softmax
};

class Activation
{
public:
    virtual ~Activation() = default;
    virtual void main(VectorXf& input, VectorXf& output) const = 0;
    virtual void prim(VectorXf& input, VectorXf& output) const = 0;
};

class Sigmoid : public Activation
{
    void main(VectorXf& input, VectorXf& output) const override;
    void prim(VectorXf& input, VectorXf& output) const override;
};

class Softmax : public Activation
{
    void main(VectorXf& input, VectorXf& output) const override;
    void prim(VectorXf& input, VectorXf& output) const override;
};

class Cost
{
public:
    virtual ~Cost() = default;
    
    virtual void gradient(const VectorXf& x, const VectorXf& y, VectorXf* result) const = 0;
};

class Quadratic : public Cost
{
public:
    Quadratic(VectorXf* derivative);
    void gradient(const VectorXf& x, const VectorXf& y, VectorXf* result) const override;
    
private:
    VectorXf* m_derivative;
};

class CrossEntropy : public Cost
{
public:
    CrossEntropy();
    void gradient(const VectorXf& x, const VectorXf& y, VectorXf* result) const override;
};
#endif /* algebra_hpp */
