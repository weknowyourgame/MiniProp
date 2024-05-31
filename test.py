from engine import Engine

# Usage/Tests for engine
x = Engine(2)  
y = Engine(3)  

# Forward pass
z = x * y 
print(z)  

# Backward pass
z.grad = 1  
z.backward()  
print(x.grad)  
print(y.grad) 
