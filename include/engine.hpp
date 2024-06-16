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

class Network
{
public:
    Network(const int sizes[], const int& N, const ActivationType& actiType, const CostType& costType);
    ~Network();
    
    static Network* loadFile(const std::string& fileName);
    
    void SGD(Dataset& dataset, const size_t& miniBatchSize, const size_t& epoch, const float& eta, const bool displayProgress = false);
    void feedForward(VectorXf& input) const;
    
    void print() const;
    void to_csv(const std::string& dest) const;
    void toBinary(const std::string& dest) const;
    void toBinary(boost::archive::binary_oarchive & ar) const;
    void loadBinary(boost::archive::binary_iarchive & ar) const;
    
    void getStats() const;
        
    float evaluateAccuracy(Dataset& dataset) const;
    
    const ActivationType activationType;
    const CostType costType;
    
    bool operator == (const Network& other) const;
private:
    // Construction
    const int m_size;
    const int* m_sizes;
    std::vector<BaseLayer*> m_layers;
    
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
