#ifndef export_hpp
#define export_hpp

#include <stdio.h>
#include <string>
#include <Eigen/Dense>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

void export_to_csv(Eigen::MatrixXf* matrix, const std::string& dest);
void export_to_csv(Eigen::VectorXf* vector, const std::string& dest);

#endif /* export_hpp */
