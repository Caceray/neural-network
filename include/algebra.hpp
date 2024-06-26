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
    virtual Activation* clone() const = 0;
    virtual void main(VectorXf& input) const = 0;
    virtual void prim(const VectorXf& activation, VectorXf& output) const = 0;
};

class Sigmoid : public Activation
{
    Activation* clone() const override;
    void main(VectorXf& input) const override;
    void prim(const VectorXf& activation, VectorXf& output) const override;
};

class Softmax : public Activation
{
    Activation* clone() const override;
    void main(VectorXf& input) const override;
    void prim(const VectorXf& activation, VectorXf& output) const override;
};

class Cost
{
public:
    virtual ~Cost() = default;
    virtual void getGradient(const VectorXf& computedOutput, const VectorXf& expectedOutput, VectorXf& result) const = 0;
};

class Quadratic : public Cost
{
public:
    Quadratic(VectorXf* derivative);
    void getGradient(const VectorXf& computedOutput, const VectorXf& expectedOutput, VectorXf& result) const override;
    
private:
    VectorXf* m_derivative;
};

class CrossEntropy : public Cost
{
public:
    CrossEntropy();
    void getGradient(const VectorXf& computedOutput, const VectorXf& expectedOutput, VectorXf& result) const override;
};
#endif /* algebra_hpp */
