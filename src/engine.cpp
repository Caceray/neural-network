#include <cmath>
#include <random>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <Eigen/Dense>

#include "export.hpp"
#include "engine.hpp"

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

using namespace std;

using Eigen::MatrixXf;
using Eigen::VectorXf;

string vectorAsString(const VectorXf& v)
{
    stringstream ss;
    ss << "[";
    for(int i(0); i<v.size(); i++)
    {
        ss << fixed << setprecision(3) << v(i);
        if(i < v.size()-1)
        {
            ss << ", ";
        }
    }
    ss << "]";
    return ss.str();
};

Network::Network(const int sizes[], const int& N, const ActivationType& actiType, const CostType& costType):
activationType(actiType),
costType(costType),
m_sizes(vector<int>(N)),
m_layers(vector<BaseLayer*>(N-1))
{
    for(int i(0); i<N-2; i++)
    {
        this->m_layers[i] = new HiddenLayer(sizes[i], sizes[i+1], ActivationType::Sigmoid);
    }
    
    this->m_layers.back() = new OutputLayer(sizes[N-2], sizes[N-1], actiType, costType);
}

Network::Network(const Network& other):
activationType(other.activationType),
costType(other.costType),
m_sizes(other.m_sizes),
m_layers(vector<BaseLayer*>(other.m_layers.size()))
{
    int i(0);
    for(BaseLayer* &l: this->m_layers)
    {
        l = other.m_layers[i]->clone();
        i++;
    }
}

Network::~Network()
{
    for(BaseLayer* l:this->m_layers)
    {
        delete l;
    }
}

void Network::SGD(const Dataset& dataset, const size_t& miniBatchSize, const size_t& epoch, const float& eta, const bool displayProgress)
{
    size_t nBatches(dataset.trainingSize()/miniBatchSize);
    cout << "Running SGD, batches count = "+to_string(nBatches) << "\n";
    
    if(displayProgress)
    {
        if(!dataset.validationSize())
        {
            throw logic_error("No validation set provided");
        }
        float acc(this->evaluateAccuracy(dataset));
        cout << "Accuracy BEFORE training : " << acc << "%.\n";
    }
    
    float coefficient(eta/miniBatchSize);
    for(size_t e(0); e < epoch; e++)
    {
        dataset.shuffle();
        for(size_t batch(0); batch<nBatches; batch++)
        {
            size_t offset(batch * miniBatchSize);
            for(size_t i(0); i < miniBatchSize; i++)
            {
                this->_backprop(dataset[i + offset]);
            }
            
            for(BaseLayer* l:this->m_layers)
            {
                l->updateWeightAndBias(coefficient);
            }
        }
    }
    
    if(displayProgress)
    {
        float acc(this->evaluateAccuracy(dataset));
        cout << "Accuracy AFTER training : " << acc << "%.\n";
    }
}

void Network::feedForward(VectorXf &input) const
{
    for(BaseLayer* l:this->m_layers)
    {
        l->feedForward(input);
    }
}

void Network::_backprop(const DataPair &datapair) const
{
    VectorXf activation(datapair.input);

    // Feedforward
    for(auto it = this->m_layers.begin(); it != this->m_layers.end(); it++)
    {
        (*it)->feedForwardAndSave(activation);
    }
    
    activation = datapair.output;

    // Backward
    auto it = this->m_layers.rbegin();
    for(size_t i(0); i<this->m_layers.size()-1; i++)
    {
        (*it)->getDelta(activation);
        (*it)->updateCost((*next(it))->getActivation());
        it++;
    }
    assert(*it == this->m_layers[0]);
    (*it)->getDelta(activation);
    (*it)->updateCost(datapair.input);
}

float Network::evaluateAccuracy(const Dataset& dataset) const
{
    // A valid output response is an output response where the index 
    // of the largest element is equal to the index of the largest
    // element in the target vector.
    
    float success(0);
    size_t idxTarget, idxComputed;
    for(size_t i(0); i<dataset.validationSize(); i++)
    {
        // Get validation data
        DataPair datapair(dataset.getTestData(i));
        
        // Feedfoward input
        VectorXf activation(datapair.input);
        this->feedForward(activation);
        
        // Compare output from feedforward and output target
        datapair.output.maxCoeff(&idxTarget);
        activation.maxCoeff(&idxComputed);
        
        // Evaluate success
        if(idxTarget==idxComputed)
        {
            success++;
        }
    }
    return 100 * success/dataset.validationSize();
}

void Network::getStats() const
{
    float means[2];
    float stds[2];
    stringstream ss;

    int i(0);
    for(BaseLayer* l:this->m_layers)
    {
        l->getStat(means, stds);
        ss << "Stats (mean/stdev) for layer " << i << endl; i++;
        ss << "Weights : " << means[0] << " / " << stds[0] << endl;
        ss << "Biases : " << means[1] << " / " << stds[1] << endl << endl;
    }
    cout << ss.str() << endl;
}

void Network::print() const
{
    int i(0);
    for(BaseLayer* l:this->m_layers)
    {
        cout << "Layer [" << i << "]" << endl; i++;
        l->print();
    }
}

void Network::to_csv(const string& dest) const
{
    int i(0);
    for(BaseLayer* l:this->m_layers)
    {
        stringstream file;
        file << dest << "layer_" << i; i++;
        l->to_csv(file.str());
    }
}

void Network::toBinary(const std::string &dest) const
{
    ofstream file(dest, ios::binary);
    boost::archive::binary_oarchive output(file);
    this->toBinary(output);
}

void Network::toBinary(boost::archive::binary_oarchive & ar) const
{
    ar << this->activationType;
    ar << this->costType;
    
    ar << this->m_sizes.size();
    for(const int& s:this->m_sizes)
    {
        ar << s;
    }
    
    for(BaseLayer* l:this->m_layers)
    {
        l->toBinary(ar);
    }
}

Network* Network::loadBinary(boost::archive::binary_iarchive &ar)
{
    ActivationType activationType; ar >> activationType;
    CostType costType; ar >> costType;
    
    size_t N; ar >> N;
    int sizes[N];
    for(int i(0); i<(int)N; i++)
    {
        ar >> sizes[i];
    }

    Network* network = new Network(sizes, (int)N, activationType, costType);
    for(BaseLayer* l:network->m_layers)
    {
        l->fromBinary(ar);
    }
    return network;
}

Network* Network::loadFile(const string &fileName)
{
    ifstream file(fileName, ios::binary);
    boost::archive::binary_iarchive input(file);
    
    return loadBinary(input);
}

bool Network::operator==(const Network& other) const
{
    if(this->m_sizes != other.m_sizes or
       this->activationType != other.activationType or
       this->costType != other.costType)
    {
        return false;
    }
    
    int i(0);
    for(BaseLayer* l:this->m_layers)
    {
        if(!l->equals(*other.m_layers[i]))
        {
            return false;
        }
        i++;
    }
    return true;
}
