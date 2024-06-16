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
m_size(N-1),
m_layers(new BaseLayer*[N-1])
{
    int *mySize = new int[N];
    for(int i(0); i<N; i++)
    {
        mySize[i] = sizes[i];
    }
    this->m_sizes = mySize;
    const int& layersCount(this->m_size);
    for(int i(0); i<layersCount-1; i++)
    {
        this->m_layers[i] = new HiddenLayer(sizes[i], sizes[i+1], actiType);

    }
    
    this->m_layers[layersCount-1] = new OutputLayer(sizes[N-2], sizes[N-1], actiType, costType);
}

Network::~Network()
{
    for(int i(0); i<this->m_size; i++)
    {
        delete this->m_layers[i];
    }
}

void Network::SGD(Dataset& dataset, const size_t& miniBatchSize, const size_t& epoch, const float& eta, const bool displayProgress)
{
    this->_initParameters(miniBatchSize, epoch, eta);
    
    size_t nBatches(dataset.trainingSize()/this->m_miniBatchSize);
    cout << "Running SGD, batches count = "+to_string(nBatches) << "\n";
    
    if(displayProgress)
    {
        if(!dataset.validationSize())
        {
            throw logic_error("No validation set provided");
        }
        this->evaluateAccuracy(dataset);
    }
    
    for(size_t e(0); e < this->m_epoch; e++)
    {
        dataset.shuffle();
        for(size_t i(0); i<nBatches; i++)
        {
            this->_updateMiniBatch(i, dataset);
        }
    }
    
    if(displayProgress)
    {
        this->evaluateAccuracy(dataset);
    }
}

void Network::feedForward(VectorXf &input) const
{
    for(int j(0); j<this->m_size; j++)
    {
        this->m_layers[j]->feedForward(input);
    }
}

void Network::_updateMiniBatch(size_t& offset, Dataset &dataset)
{
    for(size_t i(0); i<this->m_miniBatchSize; i++)
    {
        size_t idx(i + offset * this->m_miniBatchSize);
        this->_backprop(dataset[idx]);
    }
    this->_updateWeightsAndBiases();
}

void Network::_backprop(const DataPair &datapair) const
{
    const int& N(this->m_size);

    VectorXf activation(datapair.input);

    // Feedforward
    for(int i(0); i<N; i++)
    {
        this->m_layers[i]->feedForwardAndSave(activation);
    }
    
    activation = datapair.output;

    // Backward
    for(int i(0); i<N; i++)
    {
        size_t idx(N-1-i);
        this->m_layers[idx]->getDelta(activation);
        
        MatrixXf* dW(this->m_layers[idx]->getTempWeight());
        if(idx)
        {
            // Compute BP4
            this->m_layers[idx-1]->dotProductWithActivation(activation, dW);
        }
        else
        {
            *dW += activation * datapair.input.transpose();
        }
        
        // Compute left term of BP3
        this->m_layers[idx]->dotProductWithWeight(activation);
    }
}

void Network::_updateWeightsAndBiases()
{
    for(int i(0); i < this->m_size; i++)
    {
        this->m_layers[i]->updateWeightAndBias(this->m_coefficient);
    }
}

void Network::_updateWeightsAndBiasesAndEvaluateDelta(const Dataset& dataset)
{
    VectorXf before(dataset[0].output * 0), after(dataset[0].output * 0);
    int N((int)this->m_miniBatchSize);
    for(int i(0); i<N; i++)
    {
        VectorXf x(dataset[i].input);
        this->feedForward(x);
        before += x;
    }

    for(int i(0); i < this->m_size; i++)
    {
        this->m_layers[i]->updateWeightAndBias(this->m_coefficient);
    }
    
    for(int i(0); i<N; i++)
    {
        VectorXf x(dataset[i].input);
        this->feedForward(x);
        after += x;
    }
    
    cout << vectorAsString(before/N) << endl;
    cout << vectorAsString(after/N) << endl;
    cout << vectorAsString((before-after)/N) << endl;
    cout << endl;
}

float Network::evaluateAccuracy(Dataset& dataset) const
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
    
    for(int i(0); i<this->m_size; i++)
    {
        this->m_layers[i]->getStat(means, stds);
        ss << "Stats (mean/stdev) for layer " << i << endl;
        ss << "Weights : " << means[0] << " / " << stds[0] << endl;
        ss << "Biases : " << means[1] << " / " << stds[1] << endl << endl;
    }
    cout << ss.str() << endl;
}

void Network::print() const
{
    for(int i(0); i<this->m_size; i++)
    {
        cout << "Layer [" << i << "]" << endl;
        this->m_layers[i]->print();
    }
}

void Network::to_csv(const string& dest) const
{
    for(int i(0); i<this->m_size; i++)
    {
        stringstream file;
        file << dest << "layer_" << i;
        this->m_layers[i]->to_csv(file.str());
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
    ar << this->m_size;
    ar << this->activationType;
    ar << this->costType;
    
    for(int i(0); i<this->m_size+1; i++)
    {
        ar << this->m_sizes[i];
    }
    for(int i(0); i<this->m_size; i++)
    {
        this->m_layers[i]->toBinary(ar);
    }
}

void Network::loadBinary(boost::archive::binary_iarchive &ar) const
{
    for(int i(0); i<this->m_size; i++)
    {
        this->m_layers[i]->fromBinary(ar);
    }
}

Network* Network::loadFile(const string &fileName)
{
    ifstream file(fileName, ios::binary);
    boost::archive::binary_iarchive input(file);
    
    int N; input >> N;
    ActivationType activationType; input >> activationType;
    CostType costType; input >> costType;
    
    int *sizes = new int[N+1];
    for(int i(0); i<N+1; i++)
    {
        input >> sizes[i];
    }

    Network* network = new Network(sizes, N+1, activationType, costType);
    network->loadBinary(input);
    
    return network;
}

void Network::_initParameters(const size_t& miniBatchSize, const size_t& epoch, const float& eta)
{
    this->m_miniBatchSize = miniBatchSize;
    this->m_epoch = epoch;
    this->m_eta = eta;
    this->m_coefficient = eta/miniBatchSize;
}

bool Network::operator==(const Network& other) const
{
    if(this->m_size != other.m_size or
       this->activationType != other.activationType or
       this->costType != other.costType)
    {
        return false;
    }
    
    for(int i(0); i<this->m_size; i++)
    {
        if(!this->m_layers[i]->equals(other.m_layers[i]))
        {
            return false;
        }
    }
    return true;
}
