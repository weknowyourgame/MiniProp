#include "mlp.h"
#include <vector> 

MLP::MLP(const std::vector<int>& layerSizes) {
    for (size_t i = 0; i < layerSizes.size() - 1; ++i) {
        layers.emplace_back(Layer(layerSizes[i], layerSizes[i + 1]));
    }
}

std::vector<double> MLP::forward(std::vector<double> input) {
    for (int i=0; i<layers.size(); i++) {
        input = layers[i].forward(input);
    }
    return input;
}
