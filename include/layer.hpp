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
    BaseLayer(const size_t& in, const size_t& out,
              const unsigned char& activationMode);
    virtual ~BaseLayer();
    
    // Getters
    size_t getSize() const;
    MatrixXf* getTempWeight() const;
    
    // Main methods
    void feedForward(VectorXf& a) const;
    void feedForwardAndSave(VectorXf& a);
    
    void dotProductWithActivation(const VectorXf& input, MatrixXf* result) const;
    void dotProductWithWeight(VectorXf& input) const;
    void updateWeightAndBias(const float& K);
    
    // Virtual methods
    virtual void getDelta(VectorXf& a) const = 0;
    
    // Stats
    void getStat(float means[], float stds[]) const;
    void print() const;
    
    // Export
    void to_csv(const std::string& dest) const;
    
    void toBinary(boost::archive::binary_oarchive & ar) const;
    void fromBinary(boost::archive::binary_iarchive & ar);

protected:
    static std::mt19937 Generator;
    
    size_t m_inSize;
    size_t m_outSize;
    
    // Current weights and biases
    VectorXf* m_biases;
    MatrixXf* m_weights;
    
    // Temporary weights and biases
    VectorXf* m_deltaB;
    MatrixXf* m_deltaW;
    
    Activation* m_activationEngine;
    
    VectorXf m_activation;
    VectorXf m_derivative;
    
    void _applyMain(VectorXf& a, VectorXf& result) const;
    void _applyPrim(VectorXf& a, VectorXf& result) const;
    
private:
    void _initializeBuffers();
    void _getActivation(VectorXf& a) const;
};

class HiddenLayer : public BaseLayer
{
public :
    HiddenLayer(size_t& in, size_t& out,
                unsigned char activationMode=0);
    
    void getDelta(VectorXf& a) const override;
};

class OutputLayer : public BaseLayer
{
public:
    OutputLayer(const size_t& in, const size_t& out,
                const unsigned char& activationMode,
                unsigned char costMode=0);
    ~OutputLayer();
    
    void getDelta(VectorXf& a) const override;
    
private:
    Cost* m_costEngine;
};
#endif /* layer_hpp */
