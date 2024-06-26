#ifndef layer_hpp
#define layer_hpp

#include <stdio.h>
#include <random>
#include <string>
#include <Eigen/Dense>
#include "algebra.hpp"

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

using Eigen::MatrixXf;
using Eigen::VectorXf;

class BaseLayer
{
public:
    BaseLayer() = delete;
    
    BaseLayer(const int& in, const int& out, const ActivationType& actiType);
    BaseLayer(const BaseLayer& other);
    BaseLayer(BaseLayer&& other);
    
    virtual BaseLayer* clone() const = 0;
    virtual ~BaseLayer();
    
    // Main methods
    void feedForward(VectorXf& a) const;
    void feedForwardAndSave(VectorXf& a);
    void updateCost(const VectorXf& activation);
    const VectorXf& getActivation() const { return this->m_activation; }
    
    void updateWeightAndBias(const float& K);
    
    // Virtual methods
    virtual void getDelta(VectorXf& a) = 0;
    
    // Stats
    void getStat(float means[], float stds[]) const;
    void print() const;
    
    // Export
    void to_csv(const std::string& dest) const;
    
    void toBinary(boost::archive::binary_oarchive & ar) const;
    void fromBinary(boost::archive::binary_iarchive & ar);

    bool equals(const BaseLayer& other) const;
    
    const int inSize;
    const int outSize;
protected:
    static std::mt19937 Generator;
    
    // Current weights and biases
    VectorXf m_biases;
    MatrixXf m_weights;
    
    // Temporary weights and biases
    VectorXf m_deltaB;
    MatrixXf m_deltaW;
    
    Activation* m_activationEngine;
    
    VectorXf m_activation;
    VectorXf m_derivative;
    VectorXf m_deltaComputed;
    
    void _applyMain(VectorXf& a) const;
    
private:
    void _initializeBuffers();
};

class HiddenLayer : public BaseLayer
{
public :
    HiddenLayer(const int& in, const int& out, const ActivationType& actiType);
    HiddenLayer(const HiddenLayer& other):BaseLayer(other){}
    HiddenLayer(const BaseLayer& other):BaseLayer(other){}
    
    BaseLayer* clone() const override;
    void getDelta(VectorXf& product_next) override;
};

class OutputLayer : public BaseLayer
{
public:
    OutputLayer(const int& in, const int& out, const ActivationType& actiType, const CostType& costType);
    OutputLayer(const OutputLayer& other);
    OutputLayer(OutputLayer&& other) = delete;
    OutputLayer& operator=(const OutputLayer& other) = delete;
    OutputLayer& operator=(OutputLayer&& other) = delete;
    ~OutputLayer();
    
    BaseLayer* clone() const override;
    void getDelta(VectorXf& expectedOutput) override;
    
private:
    Cost* m_costEngine;
};
#endif /* layer_hpp */
