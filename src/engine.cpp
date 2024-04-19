#include <cmath>
#include <random>
#include <string>
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

Network::Network(size_t sizes[], const int& N, const NetworkActivationMode& mode):
m_size(N-1),
m_sizes(new size_t[N]),
m_layers(new BaseLayer*[N-1])
{
    for(int i(0); i<N; i++)
    {
        this->m_sizes[i] = sizes[i];
    }

    size_t& layersCount(this->m_size);
    for(int i(0); i<layersCount-1; i++)
    {
        this->m_layers[i] = new HiddenLayer(sizes[i], sizes[i+1]);

    }
    
    unsigned char charMode = static_cast<unsigned char>(mode);
    this->m_layers[layersCount-1] = new OutputLayer(sizes[layersCount-1], sizes[layersCount], charMode);
}

Network::~Network()
{
    for(int i(0); i<this->m_size; i++)
    {
        delete this->m_layers[i];
    }
}

void Network::SGD(Dataset& dataset, float& accuracy, const size_t& miniBatchSize, const size_t& epoch, const float& eta)
{
    this->_initParameters(miniBatchSize, epoch, eta);
    
    size_t nBatches(dataset.trainingSize()/this->m_miniBatchSize);
    cout << "Batches count = "+to_string(nBatches) << endl;
    
    this->evaluateAccuracy(dataset, accuracy, 0);
    for(int epoch(0); epoch < this->m_epoch; epoch++)
    {
        dataset.shuffle();
        for(int i(0); i<nBatches; i++)
        {
            this->_updateMiniBatch(i, dataset);
        }
    }
    this->evaluateAccuracy(dataset, accuracy, epoch);
}

void Network::feedForward(VectorXf &input) const
{
    for(int j(0); j<this->m_size; j++)
    {
        this->m_layers[j]->feedForward(input);
    }
}

void Network::_updateMiniBatch(int& offset, Dataset &dataset)
{
    for(int i(0); i<this->m_miniBatchSize; i++)
    {
        size_t idx(i + offset * this->m_miniBatchSize);
        this->_backprop(dataset[idx]);
    }
    this->_updateWeightsAndBiases();
}

void Network::_backprop(const DataPair &datapair) const
{
    const size_t& N(this->m_size);

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

void Network::evaluateAccuracy(Dataset& dataset, float& accuracy, const size_t& id) const
{
    float success(0);
    size_t idxTarget, idxComputed;
    for(int i(0); i<dataset.validationSize(); i++)
    {
        DataPair datapair(dataset.getTestData(i));
        VectorXf activation(datapair.input);
        this->feedForward(activation);
        datapair.output.maxCoeff(&idxTarget);
        activation.maxCoeff(&idxComputed);
        if(idxTarget==idxComputed)
        {
            success++;
        }
    }
    accuracy = 100 * success/dataset.validationSize();
    stringstream ss;
    ss << " >>> Epoch [" << id;
    ss << "] success rate [" << success << "/" << dataset.validationSize() << "]";
    cout << ss.str() << endl;
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
    output << this->m_size;
    for(int i(0); i<this->m_size+1; i++)
    {
        output << this->m_sizes[i];
    }
    for(int i(0); i<this->m_size; i++)
    {
        this->m_layers[i]->toBinary(output);
    }
}

Network::Network(const string &fileName)
{
    ifstream file(fileName, ios::binary);
    boost::archive::binary_iarchive input(file);
    size_t N;
    input >> N;
    this->m_size = N;
    this->m_sizes = new size_t[N+1];
    for(int i(0); i<N+1; i++)
    {
        input >> this->m_sizes[i];
    }
    
    // Init layers
    this->m_layers = new BaseLayer*[N];

    for(int i(0); i<N-1; i++)
    {
        this->m_layers[i] = new HiddenLayer(this->m_sizes[i], this->m_sizes[i+1]);
    }
    this->m_layers[N-1] = new OutputLayer(this->m_sizes[N-1], this->m_sizes[N], 1);
    
    stringstream ss;
    ss << "Loading [" << this->m_size << "] layers";
    
    // Fill layers
    cout << ss.str() << endl;
    
    for(int i(0); i<this->m_size; i++)
    {
        this->m_layers[i]->fromBinary(input);
    }
}

void Network::_initParameters(const size_t& miniBatchSize, const size_t& epoch, const float& eta)
{
    this->m_miniBatchSize = miniBatchSize;
    this->m_epoch = epoch;
    this->m_eta = eta;
    this->m_coefficient = eta/miniBatchSize;
}
