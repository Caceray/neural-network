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

void export_to_csv(const MatrixXf& matrix, const string& dest)
{
    ofstream outputFile(dest);
    if(outputFile.is_open())
    {
        for(int i(0); i<matrix.rows(); i++)
        {
            for(int j(0); j<matrix.cols(); j++)
            {
                outputFile << matrix(i, j);
                if (j < matrix.cols() - 1)
                {
                    outputFile << ",";
                }
            }
            outputFile << "\n";
        }
    }
    outputFile.close();
}

void export_to_csv(const VectorXf& vector, const string& dest)
{
    ofstream outputFile(dest);
    if(outputFile.is_open())
    {
        for(int i(0); i<vector.size(); i++)
        {
            outputFile << vector(i);
            outputFile << "\n";
        }
    }
    outputFile.close();
}
