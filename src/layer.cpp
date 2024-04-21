#include "layer.hpp"
#include <string>
#include <iostream>
#include "export.hpp"

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

using namespace std;

normal_distribution<float> distribution{0, 1};
mt19937 BaseLayer::Generator(10000);

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

BaseLayer::BaseLayer(const size_t& in, const size_t& out,
                     const unsigned char& activationMode):
m_inSize(in),
m_outSize(out),
m_biases(new VectorXf(out)),
m_weights(new MatrixXf(out, in)),
m_deltaB(new VectorXf(out)),
m_deltaW(new MatrixXf(out, in))
{
    this->_initializeBuffers();
    
    MatrixXf*& W(this->m_weights);
    VectorXf*& B(this->m_biases);
    
    *this->m_weights = W->unaryExpr([](float){return distribution(BaseLayer::Generator);});
    *B = B->unaryExpr([](float){return distribution(BaseLayer::Generator);});
    
    *W /= pow(in, .5);

    switch(activationMode)
    {
        case 0 :
            this->m_activationEngine = new Sigmoid();
            break;
        case 1 :
            this->m_activationEngine = new Softmax();
            break;
        default :
            throw;
    }
}

BaseLayer::~BaseLayer()
{
    delete this->m_biases;
    delete this->m_weights;
    
    delete this->m_deltaB;
    delete this->m_deltaW;
    
    delete this->m_activationEngine;
}

void BaseLayer::_initializeBuffers()
{
    this->m_deltaW->setZero();
    this->m_deltaB->setZero();
}

void BaseLayer::_applyMain(VectorXf &a, VectorXf& result) const
{
    this->m_activationEngine->main(a, result);
}

void BaseLayer::_applyPrim(VectorXf &a, VectorXf& result) const
{
    this->m_activationEngine->prim(a, result);
}

void BaseLayer::feedForward(VectorXf &a) const
{
    this->_getActivation(a);
    this->_applyMain(a, a);
}

void BaseLayer::feedForwardAndSave(VectorXf &a)
{
    this->_getActivation(a);
    this->_applyMain(a, this->m_activation);
    this->_applyPrim(a, this->m_derivative);
    a = this->m_activation;
}

void BaseLayer::_getActivation(VectorXf &a) const
{
    a = *(this->m_weights)*a + *(this->m_biases);
}

void BaseLayer::dotProductWithActivation(const VectorXf &delta, MatrixXf *dW) const
{
    *dW += delta * (this->m_activation).transpose();
}

void BaseLayer::dotProductWithWeight(VectorXf& input) const
{
    input = (*(this->m_weights)).transpose() * input;
}

void BaseLayer::updateWeightAndBias(const float &K)
{
    *(this->m_weights) -= K * *(this->m_deltaW);
    *(this->m_biases) -= K * *(this->m_deltaB);
    this->_initializeBuffers();
}

size_t BaseLayer::getSize() const
{
    return this->m_outSize;
}

MatrixXf* BaseLayer::getTempWeight() const
{
    return this->m_deltaW;
}

HiddenLayer::HiddenLayer(size_t& in, size_t& out, unsigned char activationMode):BaseLayer(in, out, activationMode){}

void HiddenLayer::getDelta(VectorXf& a) const
{
    // Equation BP2, a is left term : w^{l+1}T * d^{l+1}
    a = a.array() * this->m_derivative.array();
    *(this->m_deltaB) += a;
}

OutputLayer::OutputLayer(const size_t& in, const size_t& out,
                         const unsigned char& activationMode,
                         unsigned char costMode):BaseLayer(in, out, activationMode)
{
    switch (costMode)
    {
        case 0:
            this->m_costEngine = new Quadratic(&this->m_derivative);
            break;
        case 1 :
            this->m_costEngine = new CrossEntropy();
            break;
        default:
            throw;
    }
    
//    (*this->m_biases)(1) = 10;
}

OutputLayer::~OutputLayer()
{
    delete this->m_costEngine;
}

void OutputLayer::getDelta(VectorXf& a) const
{
    // Equation BP1
    this->m_costEngine->gradient(this->m_activation, a, &a);
    *(this->m_deltaB) += a;
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
    getStatistics(means, stds, *this->m_weights, *this->m_biases);
}

void BaseLayer::print() const
{
    cout << *this->m_weights << endl;
    cout << *this->m_biases << endl;
}

void BaseLayer::to_csv(const string& dest) const
{
    string weightsFile(dest + "_weight.csv"), biasesFile(dest + "_bias.csv");
    export_to_csv(this->m_weights, weightsFile);
    export_to_csv(this->m_biases, biasesFile);
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
    for(int i(0); i<N; i++)
    {
        ar >> v(i);
    }
}

template<class Archive>
void serializeMatrix(Archive & ar, const MatrixXf& m)
{
    size_t cols(m.cols()), rows(m.rows());
    ar << cols << rows;
    for(int col(0); col<cols; col++)
    {
        for(int row(0); row<rows; row++)
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
    for(int col(0); col<cols; col++)
    {
        for(int row(0); row<rows; row++)
        {
            ar >> m(row, col);
        }
    }
}

void BaseLayer::toBinary(boost::archive::binary_oarchive & ar) const
{
    serializeVector(ar, *this->m_biases);
    serializeMatrix(ar, *this->m_weights);
}

void BaseLayer::fromBinary(boost::archive::binary_iarchive & ar)
{
    unserializeVector(ar, *this->m_biases);
    unserializeMatrix(ar, *this->m_weights);
}
