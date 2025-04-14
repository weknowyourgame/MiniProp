#pragma once
#include <vector>
#include "neuron.h"

class Layer {
private:
    std::vector<Neuron> neurons;

public:
    Layer(int nIn, int nOut);

    std::vector<double> forward(std::vector<double>& inVector);
};
