#include <iostream>
#include "mnist.hpp"
#include "dataset.hpp"
#include "engine.hpp"

void saveAndLoad()
{
    // Initialize a network with 3 hidden layers
    const int N(5);
    size_t sizes[N] = {100, 30, 25, 20, 10};
    Network net(sizes, N, NetworkActivationMode::Sigmoid);
    
    // Export to CSV
    std::cout << "Exporting to CSV..." << std::endl;
    net.to_csv("./exports/");
    
    // Export to binary
    std::cout << "Exporting to binary..." << std::endl;
    net.toBinary("./exports/myNetwork");
    
    // Load from binary
    std::cout << "Loading from binary..." << std::endl;
    Network net2("./exports/myNetwork");
}

void trainWithMnist()
{
    Dataset dataset;
    
    std::cout << "Load MNIST dataset" << std::endl;
    MNIST::load(dataset);
    
    // Initialize the network for MNIST with 30 hidden neurons
    const int N(3);
    size_t sizes[N] = {784,30,10};
    Network net(sizes, N, NetworkActivationMode::Sigmoid);
    
    std::cout << "Run Stochastic Gradient Descent algorithm" << std::endl;
    size_t miniBatchSize(10), epoch(5);
    float eta(3), accuracy;
    net.SGD(dataset, accuracy, miniBatchSize, epoch, eta);
    
    net.toBinary("./exports/myNetwork");
}

int main(int argc, const char * argv[])
{
//    saveAndLoad();
    trainWithMnist();
}