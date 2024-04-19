#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "export.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using Eigen::MatrixXf;
using Eigen::VectorXf;

void export_to_csv(MatrixXf* matrix, const string& dest)
{
    ofstream outputFile(dest);
    if(outputFile.is_open())
    {
        for(int i(0); i<matrix->rows(); i++)
        {
            for(int j(0); j<matrix->cols(); j++)
            {
                outputFile << (*matrix)(i, j);
                if (j < matrix->cols() - 1)
                {
                    outputFile << ",";
                }
            }
            outputFile << "\n";
        }
    }
    outputFile.close();
}

void export_to_csv(VectorXf* vector, const string& dest)
{
    ofstream outputFile(dest);
    if(outputFile.is_open())
    {
        for(int i(0); i<vector->size(); i++)
        {
            outputFile << (*vector)(i);
            outputFile << "\n";
        }
    }
    outputFile.close();
}

//template<class Archive>
//void serializeVector(Archive& ar, const VectorXf& v)
//{
//    ar << v.size();
//    for(int i(0); i<v.size(); i++)
//    {
//        ar << v(i);
//    }
//}
//
//template<class Archive>
//void unserializeVector(Archive& ar, VectorXf& v)
//{
//    size_t N; ar >> N;
//    v = VectorXf(N);
//    for(int i(0); i<N; i++)
//    {
//        ar >> v(i);
//    }
//}

//template<class Archive>
//void serializeMatrix(Archive & ar, const MatrixXf& m)
//{
//    size_t cols(m.cols()), rows(m.rows());
//    ar << cols << rows;
//    for(int col(0); col<cols; col++)
//    {
//        for(int row(0); row<rows; row++)
//        {
//            ar << m(row, col);
//        }
//    }
//}
//
//template<class Archive>
//void unserializeMatrix(Archive & ar, MatrixXf& m)
//{
//    size_t cols, rows;
//    ar >> cols;
//    ar >> rows;
//    m = MatrixXf(rows, cols);
//    for(int col(0); col<cols; col++)
//    {
//        for(int row(0); row<rows; row++)
//        {
//            ar >> m(row, col);
//        }
//    }
//}

//void loadFromBinary(MatrixXf& m, const string& dest)
//{
//    ifstream file(dest, ios::binary);
//    boost::archive::binary_iarchive input(file);
//    unserializeMatrix(input, m);
//}
//
//template<class Archive>
//void exportToBinary(Archive&const MatrixXf& m)
//{
//    ofstream file(dest, ios::binary);
//    boost::archive::binary_oarchive output(file);
//    serializeMatrix(output, m);
//}
//
//void loadFromBinary(VectorXf& v, const string& dest)
//{
//    ifstream file(dest, ios::binary);
//    boost::archive::binary_iarchive input(file);
//    unserializeVector(input, v);
//}
//
//void exportToBinary(const VectorXf& v, const string& dest)
//{
//    ofstream file(dest, ios::binary);
//    boost::archive::binary_oarchive output(file);
//    serializeVector(output, v);
//}
