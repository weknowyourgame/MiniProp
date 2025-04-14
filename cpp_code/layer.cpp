#include "layer.h"

Layer::Layer(int nIn, int nOut) {
    for (int i = 0; i < nIn; ++i) {
        neurons.push_back(Neuron(nIn));
    }
}

std::vector<double> Layer::forward(std::vector<double>& inVector) {
    std::vector<double> outputs;

    for (int i = 0; i < neurons.size(); ++i) {
        outputs.push_back(neurons[i].activation(inVector));
    }

    return outputs;
}
