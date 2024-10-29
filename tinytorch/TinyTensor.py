#! /usr/bin/python3
from collections.abc import Sequence
from functools import reduce
## import numpy as np
import sys
import copy
import numbers
import operator
from MyStuff import *
from TinyArray import *
import math
import types

def opToStr(op):
  if op is None:
    return ''
  nn = str(op)
  if isinstance(op, types.BuiltinFunctionType):
    ki = nn.rfind(' ')
    if ki > 0:
      nn = nn[ki+1:-1]
  elif isinstance(op, types.LambdaType):
    ki = nn.find(' ')
    kj = nn[ki+1:].split('.')
    if len(kj) > 2:
      nn = '.'.join(kj[0:2])
  return nn

class Tensor(DebugHelper):
  '''
  Tiny pytorch.tensor (near-drop-in) replacement for educational purposes.
  s.expr is a set of predecessor Tensors that was applied to some operator to
  form THIS tensor. It's a cheap way to build a expression graph (inspired by
  micrograd package)
  '''
  multiline = False
  def __init__(s, data, *, dtype=None, device=None, requires_grad = False, pin_memory=False, label='', _op=None, I=0):
    InitDbg(s)
    s.requires_grad = requires_grad
    if isinstance(data, (list,tuple)):
      # create new Array
      s.data = Array(data, dtype=dtype)
    elif isinstance(data, Array):
      # share Array
      s.data = data
    elif isinstance(data, Tensor):
      # copy Array
      # Array clones shape, stride, and offset
      s.data = Array([k for k in data.item], data.shape, data.stride, data.offset)
    elif isinstance(data, (int, float, complex)):
      # create a 0D array
      s.data = Array(data)
    s.dtype = dtype
    s.device = device
    s.grad = None
    s._backwardRan = False
    s._backward = lambda: None
    s.op = _op
    s.label = label
    s._leaf = I == 0
    s.multiline = Tensor.multiline
    s._prev = set()
    s.setExpr()

  @property
  def is_leaf(s):
    if not s.requires_grad:
      return False
    return s._leaf
  def setExpr(s, op=None, kids=[]):
    if op or len(kids):
      s._prev |= set(kids[1:])
    if op:
      s.op = op
  @property
  def setGrad(s, initV=0):
    if s.requires_grad == True:
      if s.grad is None:
        if len(s.shape) == 0:
          s.grad = Tensor(0.0, I=1)
        else:
          s.grad = Tensor(Array([initV]*len(s.data), s.data.shape), I=1)
    return s.grad

  def requires_grad_(s):
    """ torch.requires_grad_() """
    s.requires_grad = True
    return s.setGrad
  def retain_grad(s):
    """ torch.retain_grad() """
    return s.requires_grad_()

  @property
  def isScalar(s):
    return len(s.shape) == 0
  @staticmethod
  def isBroadcastable(lhShape, rhShape):
    bigger = lhShape if len(lhShape) > len(rhShape) else rhShape
    rtnShape = [ k for k in bigger ]
    for nidx in range(-1, -min(len(lhShape), len(rhShape))-1, -1):
      lext = lhShape[nidx]
      rext = rhShape[nidx]
      if lext == 1:
        rtnShape[nidx] = rext
      elif rext == 1:
        rtnShape[nidx] = lext
      elif lext != rext:
        rtnShape = None
        break
    return rtnShape

  @property
  def shape(s):
    return s.data.shape

  def view(s, *dims):
    return Tensor(s.data.view(*dims), _op='view', I=1)

  def size(s, dim=None):
    if isinstance(dim, int):
      return s.data.shape[dim]
    return s.data.shape

  def item(s):
    if s.isScalar or s.shape == [1]:
      return s.data.data
    else: 
     raise ValueError(f'shape={s.shape}')

  def __getitem__(s, k):
    if isinstance(k, Tensor):
      k = k.data
    return Tensor(s.data.__getitem__(k), I=1)

  def __setitem__(s, k, v):
    if isinstance(k, Tensor):
      k = k.data
    if isinstance(v, Tensor):
      v = v.data
    s.data.__setitem__(k, v)

  def __len__(s):
    return MUL(*s.shape, 1, 1)

  def __str__(s):
    return hprt(f'Tensor(', s.data.to_string(multiline=Tensor.multiline), ')')

  def tolist(s):
    return s.data.unflattened

  @property
  def hasGrad(s):
    return s.requires_grad and s._backwardRan

  @property
  def T(s):
    if len(s.shape) == 1 and s.shape[0] > 0:
      return Tensor(Array(s.data, [s.shape[0],1]), _op='T', I=1)
    elif len(s.shape) > 2:
      raise ValueError(f'can not transpose a tensor of shape {s.shape}')
    Rtn = s.data.permute(1, 0)
    return Tensor(Rtn, _op='T', I=1)

  def unaryOpHelper(s, op):
    lh = s
    lh.setGrad
    rtnShape = s.shape
    assert isinstance(list(flattenArray(lh.data.data))[0], (int, float)), f'{lh.data.data=}'
    rtn = list(map(op, [lv.data for lv in  lh.data]))
    RtnRG = lh.requires_grad
    s.DBG3(Tensor.unaryOpHelper, RtnRG, 'unaryOpHelper(${rtn}, ${rtnShape}${rtnG}) = ${op} (${s} ${s_shape} ${rtnG})',
           rtn, rtnShape, ", G" if RtnRG else '',
           opToStr(op), s, s.shape)
    Rtn = Tensor(Array(rtn, rtnShape), requires_grad = lh.requires_grad, I=1)
    Rtn.setExpr(op, [ Rtn, lh ])
    return Rtn, lh

  def binaryOpHelper(s, op, other):
    lh = s
    rh = other if isinstance(other, Tensor) else Tensor(other, I=1)
    lhRG, rhRG = lh.requires_grad, rh.requires_grad
    RtnRG = lhRG or rhRG
    rhShape = rh.shape
    if lh.shape != rh.shape:
      if lh.isScalar and rh.isScalar:
        rtn = [ op(lh.data.data, rh.data.data) ]
        rtnShape = []
        other = s
      elif lh.isScalar:
        lhv = lh.item()
        rtn = [ op(lhv, rhv.data) for rhv in rh.data ]
        rtnShape = rh.data.shape
      elif rh.isScalar:
        rhv = rh.item()
        rtn = [ op(lhv.data, rhv) for lhv in lh.data ]
        rtnShape = lh.data.shape
      else:
        rtnShape = Tensor.isBroadcastable(lh.shape, rh.shape)
        if rtnShape is None:
          raise ValueError(f'this projection not handled {lh.shape=} {opToStr(op)} {rh.shape=}')
        rtnCoord = [0]*len(rtnShape)
        lhCoord = [0]*len(lh.shape)
        rhCoord = [0]*len(rh.shape)
        rtn = [0] * MUL(*rtnShape,1,1)
        for i in range(len(rtn)):
          rtn[i] = op(lh.data[lhCoord].data, rh.data[rhCoord].data)
          incrementCoord(lh.shape, lhCoord)
          incrementCoord(rh.shape, rhCoord)
          incrementCoord(rtnShape, rtnCoord)
        s.DBG2(Tensor.binaryOpHelper, 'binaryOpHelper PROJECTION ($rtnShape, nelem=$rtnl) = ($lhs, nelem=$lhl) $op ($rhs, nelem=$rhl)',
               rtnShape, MUL(*rtnShape,1,1),
               lh.shape, MUL(*lh.shape,1,1),
               opToStr(op),
               rh.shape, MUL(*rh.shape,1,1))
    else:
      if len(s.shape) > 0:
        rtn = [ op(lhv.data, rhv.data) for lhv, rhv in zip([lv for lv in lh.data],
                                                           [rv for rv in rh.data]) ]
      else:
        rtn = op(lh.data.data, rh.data.data)
      rtnShape = lh.data.shape

    rtnG = " G" if RtnRG else ""

    s.DBG2(Tensor.binaryOpHelper,
           'binaryOpHelper((${rtn}, ${rtnShape}${lG}) = (${s} ${sShape}${RG}) ${op} (${rh} ${rhShape}${RG})',
           rtn, rtnShape, " G" if RtnRG else "",
           s, s.shape, " G" if lhRG else "",
           opToStr(op), rh, rhShape,
           rG = " G" if rhRG else "")

    Rtn = Tensor(Array(rtn, rtnShape), requires_grad = RtnRG, I=1)
    if lh.label and isinstance(rh, Tensor) and rh.label:
      Rtn.label = f'{opToStr(op)}({lh.label}, {rh.label})'
    elif lh.label:
      Rtn.label = f'{opToStr(op)}({lh.label}, _)'
    elif isinstance(rh, Tensor) and rh.label:
      Rtn.label = f'{opToStr(op)}(_, {rh.label})'
    Rtn.setGrad, lh.setGrad, rh.setGrad
    Rtn.setExpr(op, [Rtn, lh, rh ])
    return Rtn, lh, rh

  def __add__(s, other):
    Rtn, lh, rh = s.binaryOpHelper(operator.add, other)
    if lh.requires_grad or rh.requires_grad:
      def _backward():
        if lh.requires_grad:
          s.DBG2(Tensor.backward, 'backward ADD lh ${lh_grad}  += ${Rtn_grad}',
                 lh.grad, Rtn.grad)
          lh.grad += Rtn.grad
          lh._backwardRan = True
        if rh.requires_grad:
          s.DBG2(Tensor.backward, 'backward ADD rh ${rh_grad}  += ${Rtn_grad}',
                 rh.grad, Rtn.grad)
          rh.grad += Rtn.grad
          rh._backwardRan = True
        Rtn._backwardRan = True
      Rtn._backward = _backward
    return Rtn

  def __mul__(s, other):
    Rtn, lh, rh = s.binaryOpHelper(operator.mul, other)
    if lh.requires_grad or rh.requires_grad:
      def _backward():
        if lh.requires_grad:
          s.DBG2(Tensor.backward, 'backward MUL LH ${lh_grad} += (${rh} * ${Rtn_grad})',
                 lh.grad, rh,  Rtn.grad)
          lh.grad += (rh * Rtn.grad)
          lh._backwardRan = True
        if rh.requires_grad:
          s.DBG2(Tensor.backward, 'backward MUL RH ${rh_grad} += (${lh} * ${Rtn_grad})',
                 rh.grad, lh,  Rtn.grad)
          rh.grad += (lh * Rtn.grad)
          rh._backwardRan = True
        Rtn._backwardRan = True
      Rtn._backward = _backward
    return Rtn

  def __pow__(s, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"

    Rtn, lh, rh = s.binaryOpHelper(operator.pow, other)
    if s.requires_grad:
      def _backward():
        assert not isinstance(other, Tensor)
        lhG = Array([ (other * lhV.data**(other-1))*lhG.data for lhV,lhG in zip([k for k in lh.data], [ l for l in Rtn.grad.data]) ], s.data.shape)
        lh.grad += Tensor(lhG, I=1)
        lh._backwardRan = True
        Rtn._backwardRan = True
        # lh.grad += (other * self.data**(other-1)) * out.grad
      Rtn._backward = _backward
    return Rtn

  def relu(s):
    op = lambda x: 0 if x < 0 else x
    Rtn, lh = s.unaryOpHelper(op)
    s.DBG2(Tensor.relu, '${Rtn} = relu(${lh})', Rtn, lh)
    if lh.requires_grad:
      backOp = lambda x,y: (x > 0) * y
      def _backward():
        # lh.grad += (Rtn.data > 0) * Rtn.grad
        lhG = Array(list(map(backOp, [x.data for x in Rtn.data],
                             [y.data for y in Rtn.grad.data])),
                    s.shape)
        s.DBG2(Tensor.backward, 'backward RELU lh.grad=${lh_grad} += RELU(${Rtn_data}, ${Rtn_grad})',
               lh.grad,  Rtn.data, Rtn.grad)
        lh.grad += Tensor(lhG, I=1)
        lh._backwardRan = True
        Rtn._backwardRan = True
      Rtn._backward = _backward
    return Rtn

  def sum(s):
    V = ADD(*[ v.data for v  in s.data], 0)
    lh = s
    Rtn = Tensor(V, requires_grad=s.requires_grad, I=1)
    Rtn.op = 'sum'
    Rtn._prev = set([lh])
    if s.requires_grad:
      lh.setGrad
      Rtn.setGrad
      def _backward():
        s.DBG2(Tensor.backward, 'backward SUM ${lhGD} += ${rGD}',
               lh.grad.data,  Rtn.grad.data)
        lh.grad += Rtn.grad
        lh._backwardRan = True
        Rtn._backwardRan = True
      Rtn._backward = _backward
    if s.label:
      Rtn.label = f'sum({s.label})'
    return Rtn

  def tanh(s):
    op = lambda x: (math.exp(2*x) - 1)/(math.exp(2*x)+1)
    Rtn, lh = s.unaryOpHelper(op)
    if s.requires_grad:
      t = Rtn.view(*Rtn.shape)
      def _backward():
        lh.setGrad
        lh.grad += ((1 - t**2) * Rtn.grad)
        lh._backwardRan = True
        Rtn._backwardRan = True
      Rtn._backward = _backward
    if s.label:
      Rtn.label = f'tanh({s.label})'
    return Rtn

  def __matmul__(s, other):
    lh0, rh0 = s, other if isinstance(other, Tensor) else Tensor(other, I=1)
    lhShape, rhShape = lh0.shape, rh0.shape
    delta =  len(lhShape) - len(rhShape)
    absDelta = abs(delta)
    lh, rh = lh0, rh0
    lhLabel = lh0.label if lh0.label else ''
    rhLabel = rh0.label if rh0.label else ''
    RtnLabel = f'@({lhLabel},{rhLabel})'
    # does not support batching yet
    # handles 2D/1D matmul subset of pytorch.Tensor
    # [a]@[a] --> [a] (dot product, resulting in 0-d Tensor)
    # [a,b]@[b,c] --> [a,c] (2d matmul, resulting 2d of [a,c])
    # [b]@[b,c] --> [1,b]@[b,c] -> [1,c]->[c] (LH matmul, 1d of [c])
    # [a,b] @ [b] --> [a,b]@[b,1] -> [a,1] ->[a] (mat*vec) 1d of [a]
    if not (((len(lhShape) == 2 and len(rhShape) == 2 and \
              lhShape[1] == rhShape[0])) or \
            (len(lhShape) == 1 and len(rhShape) == 1 and lhShape[0] == rhShape[0]) or
            (len(lhShape) == 1 and len(rhShape) == 2 and lhShape[0] == rhShape[0]) or
            (len(lhShape) == 1 and len(rhShape) == 2 and 1 == rhShape[0]) or
            (len(lhShape) == 2 and len(rhShape) == 1 and lhShape[1] == rhShape[0]) or
            (len(lhShape) == 0 and len(rhShape) <= 2 and rhShape[0] == 1)):
      raise ValueError(f'{lh.shape} is not compatible with {rh.shape} for matrix multiply')

    if delta < 0:
      lhShape = [1] * absDelta + lhShape
      lh = lh0.view(*lhShape)
      lh.requires_grad = lh0.requires_grad
    elif delta > 0:
      rhShape = rhShape + [1] * absDelta
      rh = rh0.view(*rhShape)
      rh.requires_grad = rh0.requires_grad
    elif delta == 0 and len(lhShape) == 1:
      actualShape = [ 1, 1]
      lh = lh0.view(1, lh.shape[0])
      lh.requires_grad = lh0.requires_grad
      lhShape = lh.shape
      rh = rh0.view(rh.shape[0], 1)
      rhShape = rh.shape
      rh.requires_grad = rh0.requires_grad
      delta = 1

    actualShape = [  lhShape[-2], rhShape[-1] ]
    newShape = list(filter(lambda x: x!= 1, actualShape))
    newLen = MUL(*newShape, 1, 1)
    if s.Debugging(Tensor.binaryOpHelper):
      if lhShape != lh0.shape or rhShape != rh0.shape:
        s.printer('MATMUL ORIG1 (${RtnLabel} ${newShape}<-${actualShape}) = LH:(${lhLabel} ${lhShape}<-${lhOrig}) @ RH:(${rhLabel} ${rhShape}<-${rhOrig})',
                  RtnLabel, newShape, actualShape, lhLabel, lhShape,lh0.shape, rhLabel, rhShape,rh0.shape)
      else:
        s.printer('MATMUL ORIG2 (${RtnLabel} ${newShape}<-${actualShape}) = LH:(${lhLabel}${lhShape}) @ RH:(${rhLabel} ${rhShape})',
                  RtnLabel, newShape,actualShape, lhLabel, lhShape, rhLabel, rhShape)

    RG = lh0.requires_grad or rh0.requires_grad
    Rtn = Tensor(Array([0]*newLen, actualShape), requires_grad=RG, I=1)
    for r in range(actualShape[0]):
      for c in range(actualShape[1]):
        lhR = lh[r].data
        rhC = rh[:,c].data
        V = 0
        for a, b in zip([k for k in lhR], [k for k in rhC]):
          V += a.data * b.data
        s.DBG2('matmul',
               ' matmul [${r},${c}] = DOT([${lhRShape}, ${lhRStride} ${lhROffset}], [${rhCShape}, ${rhCStride}, ${rhCOffset}]) V=${V}',
               r,c,
               lhR.shape, lhR.stride, lhR.offset,
               rhC.shape, rhC.stride, rhC.offset, V)
        s.DBG2('matmul2',' matmul [${r},${c}] = DOT(${lhR}, ${rhC}) V=${V}',
               r,c, lhR, rhC, V)
        Rtn.data[r,c] = V
    s.DBG2(Tensor.binaryOpHelper, 'MATMUL ${actualShape} = LH:(${lhShape} @ RH:(${rhShape}) newLen=${newLen}',
           actualShape, lhShape, rhShape,  newLen)
    IsBackward = False
    if RG:
      def _backward():
        IsBackward = True
        g0,g1,g2 = Rtn.setGrad, lh0.setGrad, rh0.setGrad
        RtnLabel = Rtn.label
        s.DBG2(Tensor.backward, 'backward MATMUL0 LH(${RtnLabel}${Rtn},${RtnS}) = (${LHLabel}$LH,$LHS) @ (${RHLabel}$RH,$RHS)',
               RtnLabel, Rtn, Rtn.shape, lhLabel, lh0, lh0.shape, rhLabel, rh0, rh0.shape)

        if lh0.requires_grad:
          assert lh.requires_grad, f'{lh=} {lh.shape=} {lh.data.stride}, {lh.data.offset}'
          Dst = lh0.grad.view(*lhShape)
          # assert Dst.data.item ==  lh.grad.data.item
          QQ = rh.T
          s.DBG2(Tensor.backward, 'backward MATMUL1A sizes LH.grad:((${lhLabel}).grad(${DstS})) += ($RtnLabel).grad(${RtnS}) @ rh.T(${rhLabel}${QQS}<-${rhS} ${rhOrig})',
                 lhLabel, Dst.shape, RtnLabel, Rtn.grad.shape, rhLabel, QQ.shape, rh.shape, rh0.shape)
          if Dst.shape != lh0.shape:
            Dst.data.simplifyInPlace()
            oldShape = [ k for k in Dst.shape ]
            s.DBG2(Tensor.binaryOpHelper, ' backward MATMUL 1B simplified ${oldShape}->${newShape}',
                   oldShape, Dst.shape)
          Dst += Rtn.grad.view(actualShape) @ QQ
          Dst = Dst.view(lh0.shape)
          s.DBG2(Tensor.backward, ' backward MATMUL1B LH.grad:(${lhLabel}).grad(${Dst}, ${DstS}) += (${RtnLabel}).grad(${RtnG}, ${RtnS}) @ (${rhLabel}).T(${QQ}, ${QQS})',
                 lhLabel, Dst,  Dst.shape, Rtn.label,  Rtn.grad, Rtn.grad.shape, rhLabel, QQ, QQ.shape)
          assert Dst.shape == lh0.shape
          lh0.grad = Dst
          lh0._backwardRan = True
        if rh0.requires_grad:
          assert rh.requires_grad, f'{rh=} {rh.shape=} {rh.data.stride}, {rh.data.offset}'
          Dst = rh0.grad.view(*rhShape)
          s.DBG2(Tensor.backward, 'backward MATMUL2A sizes RH.grad:(${rhLabel}${DstS}<-${rhGS}) += lh.T(${lhLabel}${QQS}<-${lhS}) @ (Rtn:${RtnLabel}).grad(${RHS})',
                 rhLabel, Dst.shape, rh0.grad.shape, lhLabel, QQ.shape, lh0.shape, RtnLabel, Rtn.grad.shape)
          assert Dst.data.item ==  rh0.grad.data.item
          QQ = lh.T
          if Dst.shape != rh0.shape:
            oldShape = [ k for k in Dst.shape ]
            Dst.data.simplifyInPlace()
            s.DBG2(Tensor.binaryOpHelper, 'backward MATMUL 2B simplified ${oldShape}->${newShape}',
                   oldShape, Dst.shape)
          Dst += QQ @ Rtn.grad.view(actualShape)
          Dst = Dst.view(*rh0.shape)
          s.DBG2(Tensor.backward, 'backward MATMUL2B RH.grad(${rhLabel}${Dst},${DstS}) += lh.T(${lhLabel}${QQ}, ${QQS}) @ (Rtn:${RtnLabel}).grad(${RH}, ${RHS})',
                 rhLabel, Dst, Dst.shape, lhLabel, QQ, QQ.shape,  RtnLabel, Rtn.grad, Rtn.grad.shape)

          rh0.grad = Dst
          assert Dst.shape == rh0.shape, f'{Dst.shape=} == {rh0.shape=}'
          lh0._backwardRan = True
        Rtn._backwardRan = True
      Rtn._backward = _backward
    Rtn.op = operator.matmul
    Rtn._prev = set([lh0, rh0])
    if len(Rtn.label) == 0:
      Rtn.label = RtnLabel

    if delta != 0:
      # because we added extra dimensions, we need to strip them out here
      oldShape = [ k for k in Rtn.shape ]
      Rtn.data.simplifyInPlace()
      s.DBG2(Tensor.binaryOpHelper, 'MATMUL simplified ${RtnLabel} ${oldShape}->${newShape}',
             RtnLabel, oldShape, Rtn.shape)
    return Rtn

  def __neg__(self): # -self
    return self * -1

  def __radd__(self, other): # other + self
    return self + other

  def __sub__(self, other): # self - other
    return self + (-other)

  def __rsub__(self, other): # other - self
    return other + (-self)

  def __rmul__(self, other): # other * self
    return self * other

  def __truediv__(self, other): # self / other
    return self * other**-1

  def __rtruediv__(self, other): # other / self
    return other * self**-1

  def __iadd__(s, o):
    if s.requires_grad == True and s.is_leaf:
      raise RuntimeError(f'a non-leaf {repr(s)} that requires grad is being used in an in-place operation.')
    s.DBG2(Tensor.binaryOpHelper, 'IADD1 ${LH} [$LHG] += ${RH} [$RHG]', repr(s), repr(s.grad), repr(o), repr(o.grad))
    Rtn, lh, rh = s.binaryOpHelper(operator.iadd, o)
    if lh.requires_grad or rh.requires_grad:
      def _backward():
        if lh.requires_grad:
          s.DBG2(Tensor.backward, 'backward IADD lh ${lh_grad}  += ${Rtn_grad}',
                 lh.grad, Rtn.grad)
          lh.grad += Rtn.grad
          lh._backwardRan = True
        if rh.requires_grad:
          s.DBG2(Tensor.backward, 'backward IADD rh ${rh_grad}  += ${Rtn_grad}',
                 rh.grad, Rtn.grad)
          rh.grad += Rtn.grad
          rh._backwardRan = True
        Rtn._backwardRan = True
      Rtn._backward = _backward
      # s._backward = _backward
    if s.requires_grad:
      return Rtn;
    s.data = Rtn.data
    Rtn.grad = s.grad
    # s.grad = Rtn.grad
    s.DBG2(Tensor.binaryOpHelper, 'IADD2 [${Rtn} ${LH}] += ${RH} [grad: $RtnG $LHG $RHG ]', repr(Rtn), repr(s), repr(o),
           repr(Rtn.grad), repr(s.grad), repr(o.grad))
    return s


  def __repr__(self):
    ## nn = self.label + ', ' if isinstance(self.label, str) and len(self.label) else ''
    oo = ', op='+opToStr(self.op) if self.op else ''
    grad = f', grad={self.grad}' if self.hasGrad else ''
    if Tensor.multiline:
      grad2 = hprt(grad=self.grad.data.to_string(multiline=True)) if self.hasGrad else ''
      return hprt('Tensor(',self.data.to_string(multiline=True),grad2,oo,')')
    else:
      return f"Tensor({self.data}{grad}{oo})"

  def __rmatmul__(self, other):
    other = Tensor(other, I=1)
    return other.__matmul__(self)

  @staticmethod
  def full(size, fill_value, *, out=None, dtype=None, layout=None, device=None, requires_grad=False, label=None):
    shape = size
    L = MUL(*shape, 1, 1)
    if L < 0:
      raise ValueError(f'illegal dimension {shape}')
    if dtype is None:
      dtype = float
    ZV = dtype(fill_value)
    A = Array([ZV] * L, shape=shape)
    if isinstance(out, Tensor):
      out.data = A
      out.device = device
      out.requires_grad = requires_grad
      out.label = label
    else:
      out = Tensor(A, dtype=dtype, device=device, requires_grad=requires_grad, label=label, I=1)
    return out

  @staticmethod
  def ones(*size, out=None, dtype=None, layout=None, device=None, requires_grad=False, label=None):
    return Tensor.full(size, 1, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, label=label)

  @staticmethod
  def zeros(*size, out=None, dtype=None, layout=None, device=None, requires_grad=False, label=None):
    return Tensor.full(size, 0, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, label=label)

  @staticmethod
  def eye(n, m=None, *, out=None, dtype=None, layout=None, device=None, requires_grad=False, label=None):
    if m is None:
      m = n
    Rtn = Tensor.full([n,m], 0, out=out, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, label=label)
    for i in range(min(*Rtn.shape)):
      Rtn[[i]*len(Rtn.shape)] = Rtn.dtype(1)
    return Rtn

  @staticmethod
  def arange(*args, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, label=None, **kwargs):
    start = 0
    end = None
    step = 1
    fail = None

    if len(args) == 1:
      end = args[0]
      if 'start' in kwargs:
        start = kwargs['start']
      if 'step' in kwargs:
        step = kwargs['step']
    elif len(args) == 2:
      start,end = args
      if 'step' in kwargs:
        step = kwargs['step']
      if 'start' in kwargs:
        fail = f'duplicate start'
    elif len(args) == 3:
      start,end,step = args
      if 'step' in kwargs or 'start' in kwargs:
        fail = f'duplicated start or step'
    else:
      fail = 'need end value'
    if dtype is None:
      dtype = type(end)
    if fail or end is None:
      raise ValueError(fail)
    L = int(math.ceil((dtype(end) - dtype(start))/dtype(step)))
    rtn = [ dtype(start) ] * L
    Shape = [L]
    for i in range(1,L):
      rtn[i] += (step*i)
    A = Array(rtn, Shape)
    if isinstance(out, Tensor):
      out.data = A
      out.device = device
      out.requires_grad = requires_grad
      out.label = label
    else:
      out = Tensor(A, dtype=dtype, device=device, requires_grad=requires_grad, label=label, I=1)
    return out

  # topological order all of the children in the graph
  def backward(s):
    topo = []
    visited = set()
    assert len(s.shape) == 0 or s.shape == [1], f'{s} must be 0-dimension (is {len(s.shape)})'
    def dfs(v, D=0):
      spc = " "*D
      if v not in visited:
        s.DBG2(Tensor.backward, '${spc}backward adding ${v} = ${opStr} ${prevSet} ',
               spc, v, opToStr(v.op), tuple(s._prev))
        visited.add(v)
        for child in v._prev:
          if isinstance(child, Tensor):
            dfs(child, D+1)
        topo.append(v)
    dfs(s)
    s.grad = Tensor.ones(*s.shape)
    s.DBG2(Tensor.backward, 'backward START ${topo}', topo)
    for i, node in reversed(list(enumerate(topo))):
      s.DBG2(Tensor.backward, ' backward (step i=${i} ${node} ${node_shape} ${isGV} op=${op})',
             i, node, node.shape, " G" if node.requires_grad else "", opToStr(node.op))
      node._backward()
