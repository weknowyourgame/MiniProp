#pragma once
#include <vector>
#include "layer.h"

class MLP {
private:
    std::vector<Layer> layers;

public:
    MLP(const std::vector<int>& layerSizes);
    std::vector<double> forward(std::vector<double> input);
};
