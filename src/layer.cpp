#include "layer.hpp"
#include <string>
#include <iostream>
#include "export.hpp"

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

using namespace std;

normal_distribution<float> distribution{0, 1};
mt19937 BaseLayer::Generator(0);

void printStatV2(MatrixXf& mat)
{
    // Calcul de la moyenne des éléments de la matrice
    float sum = mat.sum();
    float mean = sum / (mat.rows() * mat.cols());

    // Calcul de l'écart type des éléments de la matrice
    float squared_diff_sum = (mat.array() - mean).square().sum();
    float std_deviation = std::sqrt(squared_diff_sum / (mat.rows() * mat.cols()));

    // Affichage de la moyenne et de l'écart type
    cout << "Moyenne : " << mean << endl;
    cout << "Écart type : " << std_deviation << endl;
    cout << endl;
}

BaseLayer::BaseLayer(const int& in, const int& out, const ActivationType& actiType):
inSize(in),
outSize(out),
m_biases(VectorXf(out)),
m_weights(MatrixXf(out, in)),
m_deltaB(VectorXf(out)),
m_deltaW(MatrixXf(out, in))
{
    this->_initializeBuffers();
    
    MatrixXf& W(this->m_weights);
    VectorXf& B(this->m_biases);
    
    W = W.unaryExpr([](float){return distribution(BaseLayer::Generator);});
    B = B.unaryExpr([](float){return distribution(BaseLayer::Generator);});
    
    W /= pow(in, .5);

    switch(actiType)
    {
        case ActivationType::Sigmoid :
            this->m_activationEngine = new Sigmoid();
            break;
        case ActivationType::Softmax :
            this->m_activationEngine = new Softmax();
            break;
        default :
            throw;
    }
}

BaseLayer::~BaseLayer()
{
    delete this->m_activationEngine;
}

void BaseLayer::_initializeBuffers()
{
    this->m_deltaW.setZero();
    this->m_deltaB.setZero();
}

void BaseLayer::_applyMain(VectorXf &a) const
{
    a = this->m_weights * a + this->m_biases;
    this->m_activationEngine->main(a);
}

void BaseLayer::feedForward(VectorXf &a) const
{
    this->_applyMain(a);
}

void BaseLayer::feedForwardAndSave(VectorXf &a)
{
    this->_applyMain(a);
    this->m_activation = a;
    this->m_activationEngine->prim(this->m_activation, this->m_derivative);
}

void BaseLayer::updateCost(const VectorXf& activation)
{
    this->m_deltaB += this->m_deltaComputed; // BP3
    this->m_deltaW += this->m_deltaComputed * activation.transpose(); // BP4;
}

HiddenLayer::HiddenLayer(const int& in, const int& out, const ActivationType& actiType):BaseLayer(in, out, actiType){}

void HiddenLayer::getDelta(VectorXf &product_next)
{
    // Equation BP2, a is left term : w^{l+1}T * d^{l+1}
    this->m_deltaComputed = product_next.array() * this->m_derivative.array();
    product_next = this->m_weights.transpose() * this->m_deltaComputed;
}

OutputLayer::OutputLayer(const int& in, const int& out,
                         const ActivationType& actiType,
                         const CostType& costType):BaseLayer(in, out, actiType)
{
    switch (costType)
    {
        case CostType::Quadratic:
            this->m_costEngine = new Quadratic(&this->m_derivative);
            break;
        case CostType::CrossEntropy:
            this->m_costEngine = new CrossEntropy();
            break;
        default:
            throw;
    }
}

OutputLayer::~OutputLayer()
{
    delete this->m_costEngine;
}

void OutputLayer::getDelta(VectorXf& expectedOutput)
{
    // Equation BP1
    // Update expectedOutput as the same vector will be re-used during SGD
    this->m_costEngine->getGradient(this->m_activation, expectedOutput, this->m_deltaComputed);
    expectedOutput = this->m_weights.transpose() * this->m_deltaComputed;
}

void getStatistics(float means[], float stds[], const MatrixXf& W, const VectorXf& B)
{
    means[0] = W.mean();
    means[1] = B.mean();
    
    MatrixXf mat(W);
    VectorXf vec(B);
    
    stds[0] = sqrt( (mat.array() - means[0]).square().sum() / mat.size() );
    stds[1] = sqrt( (vec.array() - means[1]).square().sum() / vec.size() );
}

void BaseLayer::getStat(float means[], float stds[]) const
{
    getStatistics(means, stds, this->m_weights, this->m_biases);
}

void BaseLayer::print() const
{
    cout << this->m_weights << endl;
    cout << this->m_biases << endl;
}

void BaseLayer::to_csv(const string& dest) const
{
    string weightsFile(dest + "_weight.csv"), biasesFile(dest + "_bias.csv");
    export_to_csv(this->m_weights, weightsFile);
    export_to_csv(this->m_biases, biasesFile);
}

void BaseLayer::updateWeightAndBias(const float &K)
{
    this->m_weights -= K * this->m_deltaW;
    this->m_biases -= K * this->m_deltaB;
    this->_initializeBuffers();
}

template<class Archive>
void serializeVector(Archive& ar, const VectorXf& v)
{
    ar << v.size();
    for(int i(0); i<v.size(); i++)
    {
        ar << v(i);
    }
}

template<class Archive>
void unserializeVector(Archive& ar, VectorXf& v)
{
    size_t N; ar >> N;
    v = VectorXf(N);
    for(size_t i(0); i<N; i++)
    {
        ar >> v(i);
    }
}

template<class Archive>
void serializeMatrix(Archive & ar, const MatrixXf& m)
{
    size_t cols(m.cols()), rows(m.rows());
    ar << cols << rows;
    for(size_t col(0); col<cols; col++)
    {
        for(size_t row(0); row<rows; row++)
        {
            ar << m(row, col);
        }
    }
}

template<class Archive>
void unserializeMatrix(Archive & ar, MatrixXf& m)
{
    size_t cols, rows;
    ar >> cols;
    ar >> rows;
    m = MatrixXf(rows, cols);
    for(size_t col(0); col<cols; col++)
    {
        for(size_t row(0); row<rows; row++)
        {
            ar >> m(row, col);
        }
    }
}

void BaseLayer::toBinary(boost::archive::binary_oarchive & ar) const
{
    serializeVector(ar, this->m_biases);
    serializeMatrix(ar, this->m_weights);
}

void BaseLayer::fromBinary(boost::archive::binary_iarchive & ar)
{
    unserializeVector(ar, this->m_biases);
    unserializeMatrix(ar, this->m_weights);
}

bool BaseLayer::equals(const BaseLayer& other) const
{
    return this->m_weights == other.m_weights and this->m_biases == other.m_biases;
}
