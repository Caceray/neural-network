#ifndef dataset_hpp
#define dataset_hpp

#include <stdio.h>
#include <Eigen/Dense>
#include <vector>
#include <iostream>

using Eigen::VectorXf;

struct DataPair
{
    DataPair(const VectorXf& input, const VectorXf& output);
    DataPair(const size_t& inputDim, const size_t& outputDim);
    
    VectorXf input;
    VectorXf output;
    
    friend std::ostream& operator<<(std::ostream& os, const DataPair& datapair);
};

class Dataset
{
public:
    Dataset();
    ~Dataset();
    
    void addTrainingData(DataPair* data[], const size_t& size);
    void addValidationData(DataPair* data[], const size_t& size);
    
    size_t trainingSize();
    size_t validationSize();
    
    DataPair operator[](const size_t& i) const;
    DataPair getTestData(const size_t& i) const;
    
    void shuffle();
private:
    size_t m_sizeTraining;
    DataPair** m_training;
    
    size_t m_sizeValidation;
    DataPair** m_validation;
    
    std::vector<DataPair*> m_data;
    
    void _populate(DataPair** container, DataPair** data, const size_t& N);
};

#endif /* dataset_hpp */
