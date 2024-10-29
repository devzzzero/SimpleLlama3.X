#! /usr/bin/python3
import torch
import sys
# utilities for prettyprinting small tensors
from collections.abc import Sequence
from dataclasses import dataclass, asdict
import dataclasses as DC

def getdim(x):
  if not isinstance(x, str):
    y = str(x)
  else:
    y = x
  rows = y.split('\n')
  width = max(len(r) for r in rows)
  try:
    if isinstance(x, torch.Tensor):
      TL = len('tensor(')
      width -= TL
      for i,r in enumerate(rows):
        rows[i] = r[TL:]
      lastRow = rows[-1]
      p = lastRow.rfind(']')
      q = lastRow.rfind(',')
      P = lastRow.rfind(')')
      if p < width and p >= 0:
        width = p + 1
        rows[-1] = lastRow[:p+1]
      elif q < width and q > 0:
        # case of scalar tensors
        width = q
        rows[-1] = lastRow[:q]
      elif P < width and P > 0:
        width = P
        rows[-1] = lastRow[:P]
  except Exception as e:
    pass

  return (width, len(rows), rows)

# print(getdim(torch.eye(3,3,dtype=torch.int32)))

def vprt(*args, **kwargs):
  A = []
  for a in args:
    if isinstance(a, str):
      A.append(a)
    else:
      w,h,rows = getdim(a)
      A.extend(rows)
  for k,v in kwargs.items():
    A.append(k)
    if isinstance(v, str):
      A.append(v)
    else:
      w,h,rows = getdim(v)
      A.extend(rows)
  return '\n'.join(A)

def hprt(*args, **kwargs):
  sections = []
  for i,a in enumerate(args):
    sections.append(getdim(a))
  for k,v in kwargs.items():
    sections.append(getdim(vprt(k, v)))

  maxwidth = len(sections) -1 + sum(map(lambda f: f[0], sections))
  maxheight = max(map(lambda f: f[1], sections))
  #print(f'{maxwidth=} {maxheight=}', *map(lambda f: f[:2], sections))
  R = [[]]*maxheight
  Q = [[]]*maxheight
  #for i,s in enumerate(sections):
  #  print(f'section {i}:', s[0], s[1])
  #  print('\n'.join(s[2]))
  for r in range(maxheight):
    col = 0
    lastRow = r + 1 < maxheight
    R[r] = [' ']*maxwidth
    for i,(w,h,rows) in enumerate(sections):
      Sep = [' ']
      if i + 1 == len(sections):
        Sep = []
      if r < h:
        rl = len(rows[r])
        R[r][col:rl+col+1] = list(rows[r]) + Sep 
      col += (w + 1)
    # now pop off redundant spaces from the right
    Q[r] = ''.join(R[r]).rstrip()
    # print(f'row {r}:', Q[r])
  return '\n'.join(Q)

# getShapes(X=A, Y=B , ...)
# to print a pretty labels (X, Y, ..) of the tensors A,B...,
# then vertically followed by their shapes
# and then the tensor itself
def getShapes(**kwargs):
  return dict(zip([k for k in kwargs.keys()], list(map(lambda f: tuple(f.shape) if hasattr(f, 'shape') else '',  kwargs.values()))))

def shapes(*args, **kwargs):
  Z = getShapes(**kwargs)
  return ' '.join([ f'{tuple(V.shape)}' if hasattr(V, 'shape') else str(V) for V in args]+ [f'{K}={V}' for K,V in Z.items()])

# return a horizontally pasted view of each tensor or object, with names
# Shapes(A=A,B=B) will pretty print two tensors A and B horizontally
def Shapes(*args, **kwargs):
  Z = getShapes(**kwargs)
  R = list(map(lambda U: vprt(U), args)) + \
    list(map(lambda T: vprt(T[0], Z[T[0]], T[1]) if len(Z[T[0]]) else vprt(T[0],T[1]), kwargs.items()))
  return hprt(*R)

def pShapes(**kwargs):
  print(Shapes(**kwargs))


