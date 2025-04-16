#include "engine.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <sstream>

Engine::Engine(double data, const std::string& op, const std::vector<EnginePtr>& children)
    : data(data), grad(0), _op(op), _prev(children), _backward([](){}) {
}

EnginePtr makeEngine(double data, const std::string& op, const std::vector<EnginePtr>& children) {
    return std::make_shared<Engine>(data, op, children);
}

// Addition
EnginePtr operator+(const EnginePtr& a, const EnginePtr& b) {
    auto out = makeEngine(a->data + b->data, "+", {a, b});
    
    out->_backward = [out, a, b]() {
        a->grad += out->grad;
        b->grad += out->grad;
    };
    
    return out;
}

// Multiplication
EnginePtr operator*(const EnginePtr& a, const EnginePtr& b) {
    auto out = makeEngine(a->data * b->data, "*", {a, b});
    
    out->_backward = [out, a, b]() {
        a->grad += b->data * out->grad;
        b->grad += a->data * out->grad;
    };
    
    return out;
}

// Subtraction 
EnginePtr operator-(const EnginePtr& a, const EnginePtr& b) {
    auto out = makeEngine(a->data - b->data, "-", {a, b});
    
    out->_backward = [out, a, b]() {
        a->grad += out->grad;
        b->grad -= out->grad;
    };
    
    return out;
}

// Division 
EnginePtr operator/(const EnginePtr& a, const EnginePtr& b) {
    auto out = makeEngine(a->data / b->data, "/", {a, b});
    
    out->_backward = [out, a, b]() {
        a->grad += out->grad / b->data;
        b->grad -= out->grad * a->data / std::pow(b->data, 2);
    };
    
    return out;
}

// Power
EnginePtr pow(const EnginePtr& a, const EnginePtr& b) {
    auto out = makeEngine(std::pow(a->data, b->data), "**", {a, b});
    
    out->_backward = [out, a, b]() {
        a->grad += b->data * std::pow(a->data, b->data - 1) * out->grad;
        // Note: for the exponent gradient, we'd need logarithm, not implementing for simplicity
    };
    
    return out;
}

// ReLU
EnginePtr Engine::relu() {
    double relu_val = this->data < 0 ? 0 : this->data;
    auto out = makeEngine(relu_val, "ReLU", {this->getPtr()});
    
    out->_backward = [out, this]() {
        this->grad += (out->data > 0) ? out->grad : 0;
    };
    
    return out;
}

// Unary negation
EnginePtr operator-(const EnginePtr& a) {
    auto minus_one = makeEngine(-1);
    return minus_one * a;
}

// Backpropagation
void Engine::backward() {
    // Topological sort
    std::vector<EnginePtr> topo;
    std::set<EnginePtr> visited;
    
    std::function<void(const EnginePtr&)> build_topo = [&](const EnginePtr& v) {
        if (visited.find(v) == visited.end()) {
            visited.insert(v);
            for (const auto& child : v->_prev) {
                build_topo(child);
            }
            topo.push_back(v);
        }
    };
    
    // topological order
    build_topo(this->getPtr());
    
    // Set the gradient of the final output to 1
    this->grad = 1.0;
    
    // backpropagation in reverse topological order
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->_backward();
    }
}

std::string Engine::toString() const {
    std::stringstream ss;
    ss << "data: " << data << ", grad: " << grad;
    return ss.str();
}
