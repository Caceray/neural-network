#include "mnist.hpp"

#include <fstream>
#include <string>
#include <Eigen/Dense>
#include <vector>
#include "dataset.hpp"

#include <cassert>

using namespace std;

// Function to read integers in big-endian format
template<typename T>
T readInt(ifstream& file) {
    T value;
    file.read(reinterpret_cast<char*>(&value), sizeof(T));
    reverse(reinterpret_cast<char*>(&value), reinterpret_cast<char*>(&value) + sizeof(T));
    return value;
}

void loadBinary(const string &key, Dataset& dataset)
{
    int N = key == "train" ? 60000 : 10000;
    
    // Create output
    DataPair** data = new DataPair*[N];
    
    // Read images
    string ROOT("./data/");
    string imageFileName(ROOT + key + "-images-idx3-ubyte");
    string labelFileName(ROOT + key + "-labels-idx1-ubyte");
    ifstream imageFile(imageFileName, ios::binary);
    ifstream labelFile(labelFileName, ios::binary);
    
    if(imageFile.is_open() and labelFile.is_open())
    {
        // Read magic number : test 1
        int magicNumberImage(readInt<int>(imageFile)), magicNumberLabel(readInt<int>(labelFile));
        if(magicNumberImage!=2051 or magicNumberLabel!=2049)
        {
            throw logic_error("Invalid in magic numbers.\n");
        }
        
        // Read data size : test 2
        int numItems(readInt<int>(imageFile)), numItemsb(readInt<int>(labelFile));
        if(numItems!=numItemsb)
        {
            throw logic_error("Inconsistent count of imageFile and labelFile");
        }
        
        int numRows(readInt<int>(imageFile)), numCols(readInt<int>(imageFile));
        int size(numRows*numCols);
        
        unsigned char label;
        unsigned char image_char[size];
        
        for(int i(0); i<numItems; i++)
        {
            imageFile.read(reinterpret_cast<char*>(image_char), size);
            labelFile.read(reinterpret_cast<char*>(&label), sizeof(label));
            
            Eigen::VectorXf image(size);
            for(int j(0); j<size; j++)
            {
                image(j) = static_cast<float>(image_char[j])/255;
            }
            VectorXf output(10);
            output.setZero();
            output[label] = 1;

            data[i] = new DataPair(image, output);
        }
        
        imageFile.close();
        labelFile.close();
    }
    else
    {
        if(!imageFile.is_open() and !labelFile.is_open())
        {
            throw logic_error("Could not open files.");
        }
        else
        {
            throw;
        }
    }
    
    (key == "train") ? dataset.addTrainingData(data, N) : dataset.addValidationData(data, N);
    
    for(int i(0); i<N; i++)
    {
        delete data[i];
    }
    delete[] data;
}

void MNIST::load(Dataset& dataset)
{
    // Load training
    loadBinary("train", dataset);
    loadBinary("t10k", dataset);
}
