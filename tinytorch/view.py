from graphviz import Digraph
from Value import *
from TinyTensor import *
from MyStuff import *
import torch

def assignLabel(ss, label, requires_grad=None):
  if not isinstance(ss, torch.Tensor):
    ss.label = label
  if requires_grad is not None:
    ss.requires_grad = requires_grad

def trace(r):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(r)
  return nodes, edges
def draw_dot(r, Dir='TD'):
  dot = Digraph(format='svg', graph_attr={'rankdir' : Dir}) # TD = TOP to DOWN
  nodes, edges = trace(r)
  for n in nodes:
    uid = str(id(n))
    opstr = ''
    if isinstance(n, Value):
      dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
      opstr = opToStr(n.op)
    elif isinstance(n, Tensor):
      opstr = opToStr(n.op)
      label =  STSP0("${nLabel} | data ${nData}", n.label, n.data.to_string(multiline=True))
      if n.hasGrad:
        label += STSP0(" | grad ${nGrad}", n.grad.data.to_string(multiline=True))
      label = "{ " + label + " }"
      dot.node(name = uid, label = label, shape='record')
    if len(opstr) > 0:
      dot.node(name = uid + opstr, label = opstr, shape='record')
      dot.edge(uid + opstr, uid)
  for n1, n2 in edges:
    dot.edge(str(id(n1)), str(id(n2)) + opToStr(n2.op))
  return dot
