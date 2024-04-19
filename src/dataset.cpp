#include "dataset.hpp"
#include <vector>
#include <iomanip>
#include <Eigen/Dense>
#include <algorithm>
#include <random>

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

Dataset::~Dataset()
{
    if(this->m_sizeTraining)
    {
        for(int i(0); i<this->m_sizeTraining; i++)
        {
            delete this->m_training[i];
        }
        delete[] this->m_training;
        
        if(this->m_sizeValidation)
        {
            for(int i(0); i<this->m_sizeValidation; i++)
            {
                delete this->m_validation[i];
            }
            delete[] this->m_validation;
        }
    }
}

void Dataset::addTrainingData(DataPair **data, const size_t& size)
{
    this->m_sizeTraining = size;
    this->m_training = new DataPair*[size];
    this->_populate(this->m_training, data, size);
    
    // Add training data in vector to shuffle during SGD
    this->m_data = vector<DataPair*>(size);
    for(int i(0); i<size; i++)
    {
        this->m_data[i] = m_training[i];
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
    for(int i(0); i<size; i++)
    {
        container[i] = new DataPair(data[i]->input, data[i]->output);
    }
}

size_t Dataset::trainingSize(){return this->m_sizeTraining;}
size_t Dataset::validationSize(){return this->m_sizeValidation;}

DataPair Dataset::operator[](const size_t& idx) const
{
    return *(this->m_data[idx]);
}

DataPair Dataset::getTestData(const size_t& i) const
{
    return *(this->m_validation[i]);
}

void Dataset::shuffle()
{
    std::shuffle(m_data.begin(), m_data.end(), Generator);
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
