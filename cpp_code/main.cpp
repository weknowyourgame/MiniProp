#include <iostream>
#include "engine.h"

int main() {
    auto a = makeEngine(2.0);
    auto b = makeEngine(3.0);
    auto c = makeEngine(4.0);
    
    auto z = a * b + c;
    std::cout << "Forward pass: " << z->toString() << std::endl;
    
    z->backward();
    
    std::cout << "Gradient of a: " << a->grad << std::endl;
    std::cout << "Gradient of b: " << b->grad << std::endl;
    std::cout << "Gradient of c: " << c->grad << std::endl;
    
    // Reset gradients
    a->grad = 0;
    b->grad = 0;
    c->grad = 0;
    
    auto a2 = makeEngine(2.0);
    auto b2 = makeEngine(3.0);
    auto c2 = makeEngine(4.0);
    
    auto numerator = a2 * b2 - c2;
    auto denominator = b2 + c2;
    auto result = numerator / denominator;
    
    std::cout << "\nComplex expression result: " << result->toString() << std::endl;
    
    // backward pass
    result->backward();
    
    // Print gradients
    std::cout << "Gradient of a2: " << a2->grad << std::endl;
    std::cout << "Gradient of b2: " << b2->grad << std::endl;
    std::cout << "Gradient of c2: " << c2->grad << std::endl;
    
    // Test ReLU activation
    auto x = makeEngine(-1.0);
    auto y = makeEngine(2.0);
    
    auto activated_x = x->relu();
    auto activated_y = y->relu();
    
    std::cout << "\nReLU of -1: " << activated_x->toString() << std::endl;
    std::cout << "ReLU of 2: " << activated_y->toString() << std::endl;
    
    // Reset gradients
    y->grad = 0;
    x->grad = 0;
    
    activated_y->backward();
    std::cout << "Gradient through ReLU (positive input): " << y->grad << std::endl;
    
    // Reset activated_x gradient
    activated_x->grad = 0;
    
    activated_x->backward();
    std::cout << "Gradient through ReLU (negative input): " << x->grad << std::endl;
    
    return 0;
}
