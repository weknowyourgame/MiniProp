#include "neuron.h"

// Constructor to initialize weights and bias
Neuron::Neuron(int nin) {
    w.resize(nin);
    for (int i = 0; i < nin; i++) {
        w[i] = ((double) rand() / RAND_MAX) * 2.0 - 1.0;
    }
    b = 0.0;
}

double Neuron::activation(std::vector<double>& InVector) {
    double actRes = b;
    for (int i = 0; i < InVector.size(); i++) {
        actRes += w[i] * InVector[i];
    }
    // activate that shiii
    return relu(actRes);
}
