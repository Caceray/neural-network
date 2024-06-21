#include <iostream>
#include "mnist.hpp"
#include "dataset.hpp"
#include "engine.hpp"
#include <chrono>
#include <fstream>

void saveAndLoad()
{
    // Initialize a network with 3 hidden layers
    const int N(5);
    const int sizes[N] = {100, 30, 25, 20, 10};
    Network net(sizes, N, ActivationType::Sigmoid, CostType::Quadratic);
    
    // Export to CSV
    std::cout << "Exporting to CSV..." << std::endl;
    net.to_csv("./exports/");
    
    // Export to binary
    std::cout << "Exporting to binary..." << std::endl;
    net.toBinary("./exports/myNetwork");
    
    // Load from binary
    std::cout << "Loading from binary..." << std::endl;
    Network* net2 = Network::loadFile("./exports/myNetwork");
    if(net != *net2)
    {
        throw;
    }
    std::cout << "Test ok.\n";
}

void trainWithMnist(const ActivationType& activationType, const CostType& costType)
{
    Dataset dataset;
    
    std::cout << "Load MNIST dataset" << std::endl;
    MNIST::load(dataset);
    
    // Initialize the network for MNIST with 30 hidden neurons
    const int N(3);
    const int sizes[N] = {784,30,10};

    Network net(sizes, N, activationType, costType);
    
    std::cout << "Run Stochastic Gradient Descent algorithm" << std::endl;
    size_t miniBatchSize(10), epoch(5);
    float eta(3);
    
    auto start = std::chrono::high_resolution_clock::now();
    net.SGD(dataset, miniBatchSize, epoch, eta, true);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Elapsed time = " << duration.count()  << " ms" << std::endl;
    net.toBinary("./exports/myNetwork");
}

int main(int argc, const char * argv[])
{
    char testToRun(0);
    char trainActivationMode(0), trainCostMode(0);
    if(argc == 1)
    {
        std::cout << "Valid arguments:\n- 1 : saveAndLoad()\n- 2 : trainWithMnist()\nInput : ";
        std::cin >> testToRun;
        if(testToRun == '2')
        {
            std::cout << "Select activation mode for training:\n- 1 : Sigmoid\n- 2 : Softmax\nInput : ";
            std::cin >> trainActivationMode;
            std::cout << "Select cost mode for training:\n- 1 : Quadratic\n- 2 : Cross Entropy\nInput : ";
            std::cin >> trainCostMode;
        }
    }
    else
    {
        testToRun = *argv[1];
        if(testToRun == '2')
        {
            trainActivationMode = *argv[2];
            trainCostMode = *argv[3];
        }
    }
    
    switch(testToRun)
    {
        case '1':
            saveAndLoad();
            break;
        case '2':
            std::cout << " >>> Activation type used :";
            ActivationType actiType;
            switch(trainActivationMode)
            {
                case '1':
                    std::cout << " Sigmoid\n";
                    actiType = ActivationType::Sigmoid;
                    break;
                case '2':
                    std::cout << " Softmax\n";
                    actiType = ActivationType::Softmax;
                    break;
                default:
                    throw;
            }
            
            std::cout << " >>> Cost type used :";
            CostType costType;
            switch(trainCostMode)
            {
                case '1':
                    std::cout << " Quadratic\n";
                    costType = CostType::Quadratic;
                    break;
                case '2':
                    std::cout << " Cross Entropy\n";
                    costType = CostType::CrossEntropy;
                    break;
                default:
                    throw;
            }
            
            trainWithMnist(actiType, costType);
            break;
        default:
            throw;
    }
}
