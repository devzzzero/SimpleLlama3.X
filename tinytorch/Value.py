import math
class Value:
  def __init__(s, data, _c = (), _op = '', label = '', requires_grad=None):
    s.data = data
    s.grad = 0.0
    s._backward = lambda : None
    s._prev = set(_c)
    s.op = _op
    s.label = label
  def __repr__(s):
    return f'V(data={s.data} {s.label} {s.op})'
  # when variables are used multiple times, the grad field is overwritten
  # so the solution is to accumulate the gradients
  def __add__(s, o):
    o = o if isinstance(o, Value) else Value(o)
    out = Value(s.data + o.data, (s, o), '+')
    def _backward():
      s.grad += 1.0 * out.grad
      o.grad += 1.0 * out.grad
    out._backward = _backward
    return out
  def __neg__(s):
    return s * -1
  def __sub__(s, o):
    return s + (-o)
  def __mul__(s, o):
    o = o if isinstance(o, Value) else Value(o)
    out = Value(s.data * o.data, (s, o), '*')
    def _backward():
      s.grad += o.data * out.grad
      o.grad += s.data * out.grad
    out._backward = _backward
    return out
  def __truediv__(s, o):
    return s * (o ** -1)

  def __radd__(s, o):
    return s + o
  def __rmul__(s, o):
    return s * o
  def __rtruediv__(self, other): # other / self
    return other * self**-1

  def __pow__(s, o):
    assert isinstance(o, (int, float)), "only supporting ints and floats for now"
    out = Value(s.data ** o, (s,), f'**{o}')
    def _backward():
      s.grad += o * ((s.data ** (o - 1))) * out.grad 
    out._backward = _backward
    return out
  def exp(s):
    x = s.data
    out = Value(math.exp(x), (s,), 'exp')
    def _backward():
      s.grad += out.data * out.grad
    out._backward = _backward
    return out

  def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

    def _backward():
      self.grad += (out.data > 0) * out.grad
    out._backward = _backward

    return out

  def tanh(s):
    x = s.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x)+1)
    out = Value(t, (s,), 'tanh')
    def _backward():
      s.grad += (1 - t**2) * out.grad
      # print(f" grad:{s.grad} {t} {out.grad}")
    out._backward = _backward 
    return out
  def backward(s):
    topo = []
    visited = set()
    def dfs(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          dfs(child)
        topo.append(v)
    dfs(s)
    s.grad = 1.0
    for node in reversed(topo):
      #print('calling', node, '_backward()')
      node._backward()
    
