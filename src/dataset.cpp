#include "dataset.hpp"
#include <vector>
#include <iomanip>
#include <Eigen/Dense>
#include <algorithm>
#include <random>
#include <fstream>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

using namespace std;

using Eigen::VectorXf;

mt19937 Generator(0);

DataPair::DataPair(const VectorXf& i, const VectorXf& o):
input(i),
output(o)
{}

DataPair::DataPair(const size_t& inputDim, const size_t& outputDim):
input(VectorXf(inputDim)),
output(VectorXf(outputDim))
{
    input.setZero();
    output.setZero();
}

Dataset::Dataset():
m_sizeTraining(0),
m_sizeValidation(0)
{}

Dataset::Dataset(const string& filename):Dataset()
{
    // Check if file exists
    ifstream file(filename, ios::binary);
    if(!file.is_open())
    {
        throw logic_error("Could not open filename : "+filename);
    }
    
    // Open file
    boost::archive::binary_iarchive input(file);

    // Extract data from file
    input >> this->m_inputSize;
    input >> this->m_outputSize;
    input >> this->m_sizeTraining;
    
    this->m_training = new DataPair*[this->m_sizeTraining];
    
    for(size_t i(0); i<this->m_sizeTraining; i++)
    {
        DataPair *dp = new DataPair(this->m_inputSize, this->m_outputSize);
        for(size_t j(0); j<this->m_inputSize; j++)
        {
            input >> dp->input(j);
        }
        for(size_t j(0); j<this->m_outputSize; j++)
        {
            input >> dp->output(j);
        }
        this->m_training[i] = dp;
    }
    
    // Add training data in vector to shuffle during SGD
    this->m_data = new vector<DataPair*>(this->m_sizeTraining);
    for(size_t i(0); i<this->m_sizeTraining; i++)
    {
        (*this->m_data)[i] = m_training[i];
    }
}

Dataset::~Dataset()
{
    if(this->m_sizeTraining)
    {
        for(size_t i(0); i<this->m_sizeTraining; i++)
        {
            delete this->m_training[i];
        }
        delete[] this->m_training;
        
        if(this->m_sizeValidation)
        {
            for(size_t i(0); i<this->m_sizeValidation; i++)
            {
                delete this->m_validation[i];
            }
            delete[] this->m_validation;
        }
    }
}

void Dataset::addTrainingData(DataPair **data, const size_t& size)
{
    this->m_inputSize = data[0]->input.size();
    this->m_outputSize = data[0]->output.size();
    
    this->m_sizeTraining = size;
    this->m_training = new DataPair*[size];
    this->_populate(this->m_training, data, size);
    
    // Add training data in vector to shuffle during SGD
    this->m_data = new vector<DataPair*>(size);
    for(size_t i(0); i<size; i++)
    {
        (*this->m_data)[i] = m_training[i];
    }
}

void Dataset::addValidationData(DataPair **data, const size_t &size)
{
    this->m_sizeValidation = size;
    this->m_validation = new DataPair*[size];
    this->_populate(this->m_validation, data, size);
}

void Dataset::_populate(DataPair **container, DataPair **data, const size_t& size)
{
    for(size_t i(0); i<size; i++)
    {
        container[i] = new DataPair(data[i]->input, data[i]->output);
    }
}

size_t Dataset::trainingSize() const {return this->m_sizeTraining;}
size_t Dataset::validationSize() const {return this->m_sizeValidation;}

const DataPair& Dataset::operator[](const size_t& idx) const
{
    return *(*this->m_data)[idx];
}

DataPair Dataset::getTestData(const size_t& i) const
{
    return *(this->m_validation[i]);
}

void Dataset::shuffle() const
{
    std::shuffle(this->m_data->begin(), this->m_data->end(), Generator);
}

ostream& operator<<(ostream& os, const DataPair& datapair)
{
    const VectorXf* v(nullptr);
    
    v = &datapair.input;
    os << "Datapair : [";
    for(int i(0); i<v->size(); i++)
    {
        os << fixed << setprecision(1) << 100*(*v)[i];
        if(i<v->size()-1)
        {
            os << ", ";
        }
    }
    
    v = &datapair.output;
    os << "] / [";
    for(int i(0); i<v->size(); i++)
    {
        os << fixed << setprecision(0) << (*v)[i];
        if(i<v->size()-1)
        {
            os << ", ";
        }
    }
    os << "].";
    return os;
}

void Dataset::toBinary(const string& filename) const
{
    ofstream file(filename, ios::binary);
    boost::archive::binary_oarchive output(file);
    output << this->m_inputSize;
    output << this->m_outputSize;
    output << this->m_sizeTraining;
    for(size_t i(0); i<this->m_sizeTraining; i++)
    {
        DataPair *dp(this->m_training[i]);
        for(auto &x:dp->input)
        {
            output << x;
        }
        for(auto &x:dp->output)
        {
            output << x;
        }
    }
}
