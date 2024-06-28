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
    Network(const Network& other);
    
    Network& operator=(const Network& other) = delete;
    Network& operator=(const Network&& other) = delete;
    ~Network();
    
    static Network* loadFile(const std::string& fileName);
    static Network* loadBinary(boost::archive::binary_iarchive & ar);
    
    void SGD(const Dataset& dataset, const size_t& miniBatchSize, const size_t& epoch, const float& eta, const bool displayProgress = false);
    void feedForward(VectorXf& input) const;
    
    void print() const;
    void to_csv(const std::string& dest) const;
    void toBinary(const std::string& dest) const;
    void toBinary(boost::archive::binary_oarchive & ar) const;
    
    void getStats() const;
        
    float evaluateAccuracy(const Dataset& dataset) const;
    
    const ActivationType activationType;
    const CostType costType;
    
    bool operator == (const Network& other) const;
private:
    // Construction
    std::vector<int> m_sizes;
    std::vector<BaseLayer*> m_layers;
    
    //SGD functions
    void _backprop(const DataPair& datapair) const;
};
#endif /* engine_hpp */
