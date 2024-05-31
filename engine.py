class Engine:

    def __init__(self, data, op=(''), children=()) -> None:
        self.data = data
        self.grad = 0
        self._op = op
        self._prev = children
        
    def __repr__(self) -> str:
        return f"{self.data}, {self.grad}"
    
    def __add__(self, other):
        out = Engine(self.data + other.data, '+', (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out.backward = _backward

        print(out)
        return out

    def __mul__(self, other):
        out = Engine(self.data * other.data, '*', (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out.backward = _backward

        return out
    
    def __sub__(self, other):
        out = Engine(self.data - other.data, '-', (self, other))

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
        out.backward = _backward

        return out

    def __truediv__(self, other):     
        out = Engine(self.data / other.data, '/', (self, other))
    
        def _backward():
            self.grad += out.grad / other.data
            other.grad -= out.grad * self.data / other.data**2
        out.backward = _backward

        return out

    def __pow__(self, other):     
        out = Engine(self.data ** other.data, '**', (self, other))
    
        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out.backward = _backward

        return out

    def relu(self):
        out = Engine(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()
        
    def __neg__(self): 
        return self * -1

    def __radd__(self, other): 
        return self + other

    def __sub__(self, other): 
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other): 
        return self * other

    def __truediv__(self, other): 
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1
    