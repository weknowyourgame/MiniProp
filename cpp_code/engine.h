#pragma once
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <set>

class Engine;
using EnginePtr = std::shared_ptr<Engine>;

class Engine : public std::enable_shared_from_this<Engine> {
private:
    std::vector<EnginePtr> _prev;
    std::string _op;
    std::function<void()> _backward;

public:
    double data;
    double grad;

    // Constructor
    Engine(double data, const std::string& op = "", const std::vector<EnginePtr>& children = {});
    
    // Basic operations - changed to free functions to work with shared_ptr
    friend EnginePtr operator+(const EnginePtr& a, const EnginePtr& b);
    friend EnginePtr operator*(const EnginePtr& a, const EnginePtr& b);
    friend EnginePtr operator-(const EnginePtr& a, const EnginePtr& b);
    friend EnginePtr operator/(const EnginePtr& a, const EnginePtr& b);
    friend EnginePtr operator-(const EnginePtr& a); // Unary negation
    
    // Power operation (not an operator in C++)
    friend EnginePtr pow(const EnginePtr& a, const EnginePtr& b);
    
    // Activation function
    EnginePtr relu();
    
    void backward();

    EnginePtr getPtr() {
        return shared_from_this();
    }
    
    std::string toString() const;
};

EnginePtr makeEngine(double data, const std::string& op = "", const std::vector<EnginePtr>& children = {});

// Free function operator overloads
EnginePtr operator+(const EnginePtr& a, const EnginePtr& b);
EnginePtr operator*(const EnginePtr& a, const EnginePtr& b);
EnginePtr operator-(const EnginePtr& a, const EnginePtr& b);
EnginePtr operator/(const EnginePtr& a, const EnginePtr& b);
EnginePtr operator-(const EnginePtr& a); // Unary negation
EnginePtr pow(const EnginePtr& a, const EnginePtr& b);
