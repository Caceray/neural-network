#ifndef engine_hpp
#define engine_hpp

#include <stdio.h>
#include <random>
#include <vector>
#include <string>
#include <Eigen/Dense>

#include "algebra.hpp"
#include "dataset.hpp"
#include "layer.hpp"

using Eigen::VectorXf;

enum class NetworkActivationMode : unsigned char
{
    Sigmoid=0,
    Softmax=1
};

class Network
{
public:
    Network(const int sizes[], const int& N, const NetworkActivationMode& mode);
    Network(const std::string& fileName);
    ~Network();
    
    void SGD(Dataset& dataset, float& accuracy, const size_t& miniBatchSize, const size_t& epoch, const float& eta);
    void feedForward(VectorXf& input) const;
    
    void print() const;
    void to_csv(const std::string& dest) const;
    void toBinary(const std::string& dest) const;

    void getStats() const;
        
    void evaluateAccuracy(Dataset& dataset, float& accuracy, const size_t& id=0) const;

private:
    // Construction
    int m_size;
    const int* m_sizes;
    BaseLayer** m_layers;
    
    // SGD
    float m_eta;
    size_t m_epoch;
    float m_coefficient;
    size_t m_miniBatchSize;
    
    
    //SGD functions
    void _initParameters(const size_t& miniBatchSize, const size_t& epoch, const float& eta);
    void _backprop(const DataPair& datapair) const;
    void _updateMiniBatch(size_t& offset, Dataset& dataset);
    void _updateWeightsAndBiases();
    void _updateWeightsAndBiasesAndEvaluateDelta(const Dataset& dataset);
};
#endif /* engine_hpp */
