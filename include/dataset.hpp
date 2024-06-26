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
    Dataset(const std::string& filename);
    ~Dataset();
    
    void addTrainingData(DataPair* data[], const size_t& size);
    void addValidationData(DataPair* data[], const size_t& size);
    
    size_t trainingSize() const;
    size_t validationSize() const;
    
    const DataPair& operator[](const size_t& i) const;
    DataPair getTestData(const size_t& i) const;
    
    void shuffle() const;
    
    void toBinary(const std::string& dest) const;
    
private:
    size_t m_inputSize;
    size_t m_outputSize;
    
    size_t m_sizeTraining;
    DataPair** m_training;
    
    size_t m_sizeValidation;
    DataPair** m_validation;
    
    std::vector<DataPair*>* m_data;
    
    void _populate(DataPair** container, DataPair** data, const size_t& N);
};

#endif /* dataset_hpp */
