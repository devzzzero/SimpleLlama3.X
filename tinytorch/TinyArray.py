#! /usr/bin/python3
from collections.abc import Sequence
from functools import reduce
import operator
from MyStuff import *
from tensorhelpers import *


def getShapeOf(x, Depth=0):
  '''
  returns the shape of a multidimensional list as a list of numbers
  '''
  Dims = []
  if isinstance(x, list):
    D0 = len(x)
    Dims.append(D0)
    D1 = [ getShapeOf(d1, Depth+1) for d1 in x ]
    for ii, dS in enumerate(D1[1:]):
      if dS != D1[0]:
        raise ValueError(f'at dim {Depth} size mismatch at pos {ii+1} {dS} {D1[0]}')
    Dims.extend(D1[0])
  return Dims


def getStrideOf(*shape):
  if len(shape) == 1:
    if isinstance(shape[0], list) or isinstance(shape[0], tuple):
      shape = shape[0]
    elif isinstance(shape[0], int):
      return [1]
    else:
      raise ValueError(f'{shape=} not a list or tuple of numbers')
  if len(shape) == 0:
    return []
  if len(shape) < 1 or len(shape) != ADD(*[ isinstance(x, int) for x in shape], 0):
    raise ValueError(f'{shape=} not a list or tuple of numbers')
  DbgPrt(getStrideOf, f'{shape=}')
  Stride = list(shape[1:]) + [1]
  DbgPrt(getStrideOf, f'{Stride=}')
  for k in range(len(Stride)-1, 0,-1):
    Stride[k-1] *= Stride[k]
    DbgPrt(getStrideOf, f'{k=} {Stride=}')
  return Stride

def flattenArray(something, dtype=None):
  if isinstance(something, (list, tuple, set, range)):
    for sub in something:
      yield from flattenArray(sub, dtype)
  else:
    if dtype:
      yield dtype(something)
    else:
      yield something
def getOffsetOf(Stride, Coords, Offsets=None):
  Rtn = None
  Offset=0
  if Offsets is None:
    Offsets = [0] * (1+len(Stride))
  else:
    Offset=Offsets
  Final = Offsets[-1]
  Rtn = reduce(lambda x,y: x+y, [k*(v+o) for k, v, o in zip(Stride, Coords, Offsets)], Final)
  DbgPrt(getOffsetOf, 'getOffsetOf(Stride=$Stride, Coords=${Coords}, Offset=${Offset})=${Rtn}',
         Stride, Coords, Offset, Rtn)
  return Rtn

def unFlattenArray(shape, stride, offset, arr, dtype = None):
  DbgPrt(unFlattenArray, f'unFlattenArray({shape=}, {stride=}, {offset=})')
  def unFlattenArray2(shape, stride, offset, offset1, coord, arr, depth, dtype=None):
    SPC = '  '*depth
    DbgPrt(unFlattenArray, f'{SPC}unFlattenArray2({shape=}, {stride=}, {offset=} {offset1=} {coord=})')
    rtn = [None]*shape[0]
    if len(shape) == 1:
      idxs = [ getOffsetOf(stride, [idx], offset) + offset1 for  idx in range(shape[0]) ] 
      if IsDebugging('unFlattenArray2'):
        DbgPrt(unFlattenArray, f'{SPC}  unFlattenArray2({idxs=})')
      if dtype:
        rtn = [dtype(arr[idx]) for idx in idxs ]
      else:
        rtn = [arr[idx] for idx in idxs ]
    else:
      for idx in range(shape[0]):
        coord[0] = idx
        currOffset = getOffsetOf(stride, coord, offset) + offset1 - offset[-1]
        rtn[idx] = unFlattenArray2(shape[1:], stride[1:], offset[1:], currOffset, coord[1:], arr, depth+1, dtype)
        if IsDebugging('unFlattenArray2'):
          DbgPrt(unFlattenArray, f'{SPC}  unFlattenArray2A({idx=} {coord=} {currOffset=})->{rtn[idx]=}')
      offset[-1] = offset1
    return rtn
  coord = [0] * len(shape)
  return unFlattenArray2(shape, stride, offset, 0, coord, arr, 1, dtype)


def incrementCoord(shape, coord):
  # Treat coord as a len(s.shape) number, with each coord[i] is in base s.shape[i]
  # Simply put - Increment Coord[-1], and carry over to Coord[-2] if it overflows.
  # Returns the next coordinate, a flag that says which digit wrapped to 0
  # or -1 if the entire coord was reset to 0
  dim = len(coord)-1
  if len(shape) == 0 or len(coord)==0:
    return [],-1
  coord[dim] += 1
  while dim > 0 and coord[dim] >= shape[dim]:
    coord[dim-1] += 1
    coord[dim] = 0
    dim-=1
  if coord[0] >= shape[0]:
    coord[0] = 0
    dim-=1
  return coord,dim

## a helper class that maintains a flat list of elements
## also supports sharing of a larger arrays
## we break up the Array accessor functionality from Tensor, simply
## to keep file size lower

class Array(DebugHelper):
  def __init__(s, item, shape=None, stride=None, offset=None, dtype=None):
    InitDbg(s)
    # first copy the shape, stride, and offset, if they are supplied
    if shape is not None:
      shape = [ k for k in shape ]
    if stride is not None:
      stride = [ k for k in stride ]
    if offset is not None:
      offset = [ k for k in offset ]

    if isinstance(item, Array):
      # SHARES the stored values
      s.item = item.item
      if shape is None:
        shape = [k for k in item.shape ]
      if stride is None:
        stride = [ k for k in item.stride ]
      if offset is None:
        offset = [ k for k in item.offset ]
    elif isinstance(item, (list, tuple, set, range)):
      if shape is None:
        shape = getShapeOf(list(item))
      if stride is None:
        stride = getStrideOf(shape)
      if offset is None:
        offset = [0]*(1+len(shape))
      s.item = list(flattenArray(item, dtype))

    elif isinstance(item, (int, float, complex)):
      # a zero-D number, but we always represent it as an Array (duh!)
      # in other words, shape=[1,1,...] always simplifies to shape=[]
      if dtype:
        s.item = [ dtype(item) ]
      else:
        s.item = [ item ]
      shape = []
      stride = []
      offset = [0]
    else:
      raise ValueError(f'illegal array {item} {shape} {stride} {offset}')
    s.shape, s.stride, s.offset, s.length  = shape, stride, offset, MUL(*shape,1,1) if len(shape) else 0

  def convert(s,dtype=None):
    if dtype:
      return Array([ dtype(k) for k in s.item ], s.shape, s.stride, s.offset)
    return Array([ k for k in s.item ], s.shape, s.stride, s.offset)

  def permute(s, *dims):
    if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
      dims = dims[0]
    kkDims = set(dims)
    okDims = set(range(len(s.shape)))
    if okDims != kkDims:
      raise ValueError(f'{dims=} is not a valid pemutation of [0..{len(s.shape)-1}]')
    newShape = [ k for k in s.shape ]
    newStride = [ k for k in s.stride ]
    newOffset = [ k for k in s.offset ]
    swapped = set()
    for dim1, dim2 in enumerate(dims):
      newShape[dim1] = s.shape[dim2]
      newStride[dim1] = s.stride[dim2]
      newOffset[dim1] = s.offset[dim2]
    s.DBG2(Array.permute, 'permute(${dims}) shape:${oldShape}->${newShape} stride:${oldStride}->${newStride} offset:${oldOffset}->${newOffset}',
           dims, s.shape, newShape, s.stride, newStride, s.offset, newOffset)
    return Array(s, newShape, newStride, newOffset)

  class Iter(DebugHelper):
    ''' elementwise iteration over the tensor -- note this is DIFFERENT
    from how pytorch.tensor does it
    '''
    def __init__(s, A):
      InitDbg(s)
      s.arr = A
      s.coord = [0] * len(A.shape)
      s.code = 0
    def __iter__(s):
      return s
    def __next__(s):
      if (s.code == -1):
        raise StopIteration()
      rtn = s.arr[s.coord]
      fmt = ''
      if s.Debugging(Array.Iter):
        fmt = f'accessing (shape:{s.arr.shape} stride:{s.arr.stride} offset:{s.arr.offset} {s.coord}={rtn}'
      (newcoord, s.code) = incrementCoord(s.arr.shape, s.coord)
      s.DBG2(Array.Iter, 'Iter:${fmt} newcoord=${coord}', fmt, s.coord)
      return rtn

  def __iter__(s):
    return Array.Iter(s)

  def __len__(s):
    return s.length

  def view(s, *dims):
    ''' alllowed to have ONE -1'''
    if len(dims) == 1:
      if isinstance(dims[0], (list,tuple)):
        dims = dims[0]
    dims = list(dims)
    theLen = MUL(*s.shape, 1,1)
    newDims = list(filter(lambda x: x != -1, dims))
    currLen = MUL(*newDims, 1, 1)
    rem = theLen % currLen
    if rem or currLen > theLen or (len(dims) - len(newDims))  > 1 or theLen % currLen != 0:
      raise ValueError(f'{dims} does not divide into {s.shape}')
    newDim = theLen // currLen
    for dim, extent in enumerate(dims):
      if extent == -1:
        dims[dim] = newDim
        break
    # now have to distribute the stride and offsets!
    # if s.stride was not monotonically decreasing
    ML = list(map(lambda x: x[0]>x[1],  zip(s.stride, s.stride[1:])))
    IsDecreasing = reduce(lambda x,y: x and y, ML, False)
    NonZeroOffsets = 0 != reduce(lambda x,y: x | y, s.offset, 0)
    if (IsDecreasing or NonZeroOffsets) and len(ML)>0:
      raise ValueError(f'unsupportred {ML=} stride={s.stride} and offset={s.offset}')

    newStride = getStrideOf(*dims)
    C0 = [0] * len(s.shape)
    C1 = [0] * len(dims)
    newOffset = s.offset[0:len(dims)+1]
    if len(dims) > len(s.shape):
      newOffset.extend([0]*(len(dims) - len(s.shape)))
    s.DBG2(Array.view,'view(${repr_s} (${s_shape}, ${s_stride}, ${s_offset}) -> ${dims} (${newDims} ${newStride} ${newOffset})',
           s, s.shape, s.stride, s.offset, dims, newDims, newStride, newOffset)
    return Array(s, dims, newStride, newOffset)

  def getitemHelper(s, i):
    '''returns (item, newShape, newStride, newOffset)'''
    RemoveDim = []
    if isinstance(i, int):
      if len(s.shape) == 1:
        newOffset = [s.offset[-1]+(i*s.stride[0])]
        s.DBG2(Array.getitemHelper, '${s}.getitemHelper1(${i}, idx=${offset}', s, i, newOffset)
        return (s.item, [], [], newOffset)
      else:
        # return a len(shape)-1 dim Array
        Coords = [i] + [0]*(len(s.stride)-1)
        baseOffset = getOffsetOf(s.stride, Coords, s.offset)
        shape0 = s.shape[1:]
        stride0 = s.stride[1:]
        offset0 = s.offset[1:]
        offset0[-1] += baseOffset
        s.DBG2(Array.getitemHelper, '${s}.getitemHelper2(${i}) = (${shape},${stride},${offset})',
               s, i, offset0)
        return (s, shape0, stride0, offset0)
    else:
      newShape = [k for k in s.shape ]
      newStride = [k for k in s.stride ]
      newOffset = [k for k in s.offset ]
      s.DBG2(Array.getitemHelper, '${s}.getitemHelper3(${i}) sh:${newShape} st:${newStride} of:${newOffset}',
             s, i,
             newShape, newStride, newOffset)
      if isinstance(i, (list, tuple)):
        for dim, idx in enumerate(i):
          idxs = idx
          if isinstance(idx, int):
            newShape[dim] = 1
            newOffset[dim] += idx
            RemoveDim.append(dim)
          elif isinstance(idx, slice):
            idxs = idx.indices(newShape[dim])
            newShape[dim] = (idxs[1] - idxs[0])//idxs[2]
            newOffset[dim] += idxs[0]
            newStride[dim] *= idxs[2]
            if newShape[dim] == 1:
              RemoveDim.append(dim)
          s.DBG2(Array.getitemHelper, ' getitem3a(dim=${dim}, idx=${idx} idxs=${idxs}) -> sh:${newShape} st:${newStride} of:${newOffset}',
                 dim, idx, idxs, newShape, newStride, newOffset)

      elif isinstance(i, slice):
        # result is len(s.shape) Array with first dimension being shrunk
        idxi = i.indices(s.shape[0])
        s.DBG2(Array.getitemHelper, ' getitem3b(${i}, ${idxi})', i, idxi)
        dim = 0
        newShape[dim] = (idxi[1] - idxi[0])//idxi[2]
        newOffset[dim] += idxi[0]
        if newShape[dim] == 1:
          RemoveDim.append(dim)
        s.DBG2(Array.getitemHelper, '${s}.getitemHelper3b(${i}) sh:${newShape} st:${newStride} of:${newOffset}',
               s, i,
               newShape, newStride, newOffset)
      else:
        raise ValueError(f'{i=} is not accessible')
      for dim, extent in reversed(list(enumerate(newShape))):
        # check for degenerate case
        if extent <= 0:
          newOffset[0:dim] = [0]*dim
      # remove redundant dims in the array
      return s.simplify_(RemoveDim, i, newShape, newStride, newOffset)

  def simplify_(s, RemoveDim, i=[], newShape=None, newStride=None, newOffset=None):
    '''
    remove the following dimenstion from the array
    returns the tuple (s, newShape, newStride, newOffset)
    '''
    if len(RemoveDim) == 0:
      return (s, newShape, newStride, newOffset)
    if newShape is None:
      newShape = [ k for k in s.shape ]
    if newStride is None:
      newStride = [ k for k in s.stride ]
    if newOffset is None:
      newOffset = [ k for k in s.offset ]

    if set(RemoveDim) == set(newShape):
      Offset = reduce(lambda x,y: x+y,
                      map(lambda x: x[0]*x[1], zip(newStride + [1], newOffset)))

      s.DBG2(Array.simplify_, 'simplifyEnd: reducing to a zero dim  [${sShape} ${sStride} ${sOffset}] [${i}]  sh:${newShape} st:${newStride} of:${newOffset} ${Offset} starts at ${arrAt}',
             s.shape, s.stride, s.offset, i, newShape, newStride, newOffset, Offset, s.item[Offset])
      newShape = []
      newStride = []
      return (s, newShape, newStride, [ Offset ])
    for dim in reversed(sorted(RemoveDim)):
      skipDim = dim
      skipShape,skipOffset,skipStride = newShape[dim], newOffset[dim], newStride[dim]
      baseOffset = skipOffset*skipStride

      newShape = newShape[:dim] + newShape[dim+1:]
      newOffset = newOffset[:dim] + newOffset[dim+1:]
      newStride = newStride[:dim] + newStride[dim+1:]

      # s.DBG3(Array.simplify_, s.Debugging('verbose'), 'simplify: RemovDim:${dim} resulted in ns:${newShape} st:${newStride} of:${newOffset}',
      #        dim, newShape, newStride, newOffset)

      if len(newOffset):
        newOffset[-1] += baseOffset
      else:
        raise ValueError(f'unexpected result while removing {dim=} - {baseOffset=} {newShap=} {newStride=} {newOffset=}')
        newOffset = [baseOffset]

    s.DBG2(Array.simplify_, 'simplifyEnd: [${sShape} ${sStride} ${sOffset}] [${i}] results in RD:${RemoveDim} ns:${newShape} st:${newStride} of:${newOffset} starts at ${arrAt}',
             s.shape, s.stride, s.offset, i, RemoveDim, newShape, newStride, newOffset, s.item[newOffset[-1]])
    return (s, newShape, newStride, newOffset)

  def simplifyInPlace(s):
    ''' remove any excess shape of '1'
    '''
    RemoveDim = list(filter(lambda dim: s.shape[dim] == 1, range(len(s.shape))))
    if len(RemoveDim):
      t, s.shape, s.stride, s.offset = s.simplify_(RemoveDim)
      assert t == s
    return s

  def simplify(s):
    ''' remove any excess shape of '1'
    '''
    RemoveDim = list(filter(lambda dim: s.shape[dim] == 1, range(len(s.shape))))
    if len(RemoveDim):
      t, newShape, newStride, newOffset = s.simplify_(RemoveDim)
      assert t == s
    return Array(s, newShape, newStride, newOffset)

  def __getitem__(s, i):
    '''
    final = 0
    for d in range(len(s.shape)):
      final += offset[d] + stride[d]*i[d]
    like PyTorch, does not support back-to-front indexing via negative offsets via slice
    '''
    item, newShape, newStride, newOffset = s.getitemHelper(i)
    return Array(item, shape=newShape, stride=newStride, offset=newOffset)

  def __setitem__(s, k, v):
    LH = Array(*s.getitemHelper(k))
    item, lhShape, lhStride, lhOffset =  LH.item, LH.shape, LH.stride, LH.offset
    s.DBG2(Array.__setitem__, 'setitem(${k})=${v} {shape=${lhShape} stride=${lhStride} offset=${lhOffset}}',
           k, v, lhShape, lhStride, lhOffset)

    if isinstance(v, Array):
      rhShape = v.shape
      rhStride = v.stride
      rhOffset = v.offset
      rhItem = v.item
    elif isinstance(v, (list, tuple)):
      rhShape = getShapeOf(v)
      rhStride = getStrideOf(rhShape)
      rhOffset = [0] * len(rhShape)
      rhItem = list(flattenArray(v))
    elif isinstance(v, (int, float, complex)):
      rhShape = []
      rhStride = []
      rhOffset = (0,)
      rhItem = [v]
    else:
      raise ValueError(f'{v} is not a legal Array assignable value')
    if rhShape != lhShape:
      raise ValueError(f'{k} denotes {lhShape} but {v=} has shape={rhShape}')
    # the shapes maybe the same but the stride and offset may not be the same
    if len(lhShape) == 0:
      if isinstance(k, int):
        s.item[k + s.offset[-1]] = v
        s.DBG2(Array.__setitem__, ' setitem 1D set  ${item_k}  == ${v}', item[k], v)
      else:
        s.item[lhOffset[-1]] = v
        s.DBG2(Array.__setitem__, ' setitem 1D set [${k} ${lhOffset}] ${s_item}  == ${v}',
               k, lhOffset, s.item[lhOffset[-1]], v)
    else:
      Coord = [0] * len(rhShape)
      Zero =  [0] * len(rhShape)
      while True:
        # so idx for identical coord may be different for rh and lh
        # Coords = f'{Coord}'
        rhIdx = getOffsetOf(rhStride, Coord, rhOffset)
        lhIdx = getOffsetOf(lhStride, Coord, lhOffset)
        s.DBG2(Array.__setitem__, ' setitem ${Coord} item[${lhIdx}] <- rhItem[${rhIdx}] val=${val}',
               Coord, lhIdx, rhIdx, val = rhItem[rhIdx])

        item[lhIdx] = rhItem[rhIdx]
        # since the shape is the same, we can use a single Coord
        incrementCoord(lhShape, Coord)
        if Coord == Zero:
          break

  @property
  def repr(s):
    return f'shape:{s.shape},stride:{s.stride},offset:{s.offset}'

  @property
  def data(s):
    if s.shape == [] or s.shape == [1]:
      return s.item[s.offset[-1]]
    else:
      return unFlattenArray(s.shape, s.stride, s.offset, s.item)

  @property
  def unflattened(s):
    if s.shape == []:
      return s.item[s.offset[-1]]
    else:
      return unFlattenArray(s.shape, s.stride, s.offset, s.item)
    
  def __repr__(s):
    if s.shape == []:
      return str(s.item[s.offset[-1]])
    else:
      return str(s)

  def __str__(s):
    return s.to_string()

  def to_string(s, multiline=False):
    Sep = '' if multiline else ' '
    Coord = [0]*len(s.shape)
    Zeros = [0]*len(s.shape)
    Len = MUL(*s.shape,1, 1)
    inDim = 0
    Str = []
    inCol = 0
    if len(s.shape) == 0:
      if not isinstance(s.item, (list,tuple)):
        return str(s.item)
      return str(s.item[s.offset[0]])
    for dim, V in enumerate(Coord):
      if V == 0:
        Str.append('[')
        inDim = dim
        inCol = dim+1
    while True:
      o0 = getOffsetOf(s.stride, Coord, s.offset)
      val = s.item[o0]
      if isinstance(val, (float, complex)):
        Str.append(f'{val:.4f}')
      else:
        Str.append(str(val))
      if multiline:
        Str.append(' ')
      Coord, chDim  = incrementCoord(s.shape, Coord)
      delta = abs(chDim - inDim)
      if chDim != inDim:
        if multiline:
          Str.pop() # pop the last space
        Str.extend([']'] * delta)
        inCol -= delta
        inDm = chDim
        if chDim != -1:
          if multiline:
            Str.extend(['\n'] + [' ']*inCol)
          Str.extend(['[']*delta)
          inCol += delta
      if Coord == Zeros:
        break
    return Sep.join(Str)
    


