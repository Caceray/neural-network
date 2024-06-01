# neural-network

Implementation of SGD algorithm for dense neural network.

## Table of Contents

- [Project Name](#project-name)
  - [Table of Contents](#table-of-contents)
  - [About](#about)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)

## About

Provide a simple interface to create a neural network

## Getting Started

Instructions on setting up the project on a local machine.

### Prerequisites

- [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)
- Boost
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)

### Installation
1) Copy local repository
```bash
git clone https://github.com/Caceray/neural-network.git
```


2) Before running the project, make sure to download the MNIST dataset files and save them locally in the "data" directory within the project repository. You can download the MNIST dataset from the following link:

[Download MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

Once downloaded, extract the files and save them in the "data" directory. The directory structure should look like this:
```
neural-network/
│
├── data/
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
```

3) Use Makefile

### Usage
The library lib/neuralnetwork.a is created by Makefile, you can import this file to any other project.
