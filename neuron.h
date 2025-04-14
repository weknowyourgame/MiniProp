#pragma once

#include <vector>
#include <cstdlib>
#include <algorithm>

class Neuron {
private:
    std::vector<double> w;
    double b;

    double relu(double x) {
        return std::max(0.0, x);
    }

public:
    Neuron(int nin);
    double activation(const std::vector<double>& InVector);
};
