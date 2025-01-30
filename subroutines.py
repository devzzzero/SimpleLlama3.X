import re
import datetime
import parsedatetime as pdt # # pip install parsedatetime
import json
# from dateutil.tz import tzutcm
import sys
import os.path as path
import platform
import datetime as DT
import json as JSONR
from sortedcollections import *
from tinytorch.tensorhelpers import *
import dataclasses as DC
from dataclasses import dataclass
from typing import Literal, Tuple, List, Optional
import numbers

def findByName(name, LOCALS, GLOBALS):
  if isinstance(name, str):
    names = name.split('.')
    if len(names) == 1:
      # print(f'looking up {name=}')
      if name in LOCALS:
        return LOCALS[name]
      if name in GLOBALS:
        return GLOBALS[name]
      # print(f'looking up {name=} failed')
    else:
      start = None
      if names[0] in locals():
        start = LOCALS[names[0]]
      elif names[0] in globals():
        start = GLOBALS[names[0]]
      if start is not None:
        for name in names[1:]:
          if hasattr(start, name):
            start = getattr(start, name)
          else:
            # print(f'failed with {name=}')
            return None
        return start
    #print(f'did not find {names=}')
    return None
  return name

def getDataClassFieldNames(ds0):
  fieldNames = list(map(lambda x: x.name, DC.fields(ds0)))
  return fieldNames

def splitAndConvert(S, toType, sep=','):
  return list(map(toType, S.split(sep)))

def splitByWhiteSpace(x):
  return list(filter(lambda x: len(x) > 0, map(lambda x: x.strip(), x.splitlines())))

def convertToBool(S):
  if isinstance(S, numbers.Number):
    return bool(int(S))
  elif isinstance(S, str):
    if S.startswith("1") or S.casefold() == 'true':
      return True
    elif S.startswith("0") or S.casefold() == 'false':
      return False
  return False

def buildCmdLineParserFrom(dct, argPrefix='--', scopeSep='.',listSep=','):
  FN = OrderedDict([[ f.name, f] for f in DC.fields(dct) ])
  Rtn = []
  for k, v in FN.items():
    field = k
    arg = f'{argPrefix}{k}'
    if v.type == int or v.type == float or v.type == str:
      converter = v.type
      Rtn.append((field, arg, converter))
    elif v.type == bool:
      converter = convertToBool
      Rtn.append((field, arg, converter))
    elif v.type == List[int] or v.type == List[float] or v.type == List[str]:
      containedType = v.type.__args__[0]
      converter = lambda S: splitAndConvert(S, containedType, listSep)
      Rtn.append((field, arg, converter))
    elif DC.is_dataclass(v.type):
      partial = buildCmdLineParserFrom(v.type, argPrefix, scopeSep, listSep)
      for (subfield, subarg, converter) in partial:
        newField = f'{field}{scopeSep}{subfield}'
        arg = f'{argPrefix}{newField}'
        Rtn.append((newField, arg, converter))
    else:
      pass
      ## uhoh! can't handle this type!
  return Rtn

def getDCField(Orig, fieldName, scopeSep='.'):
  chain = fieldName.split(scopeSep)
  if len(chain) == 1:
    return getattr(Orig, fieldName)
  attrs = [ Orig, getattr(Orig, chain[0]) ]
  for subkey in chain[1:]:
    attrs.append(getattr(attrs[-1], subkey))
  return attrs[-1]

def setDCField(Orig, fieldName, Value, scopeSep='.'):
  chain = fieldName.split(scopeSep)
  if len(chain) == 1:
    kwargs = dict([[fieldName, Value]])
    setattr(Orig, fieldName, Value)
    return Orig
  else:
    # ex:
    # fieldName=foo.bar.baz, Value=3
    # chain = [ foo, bar, baz ]
    # attrs = [ Orig, Orig.foo, Orig.foo.bar, Value=3 ]
    #  attrs[2] = DC.replace(attrs[2], chain[2]=attrs[3] )
    #  attrs[1] = DC.replace(attrs[1], chain[1]=attrs[2] )
    #  attrs[0] = DC.replace(attrs[0], chain[0]=attrs[1] )
    attrs = [ Orig, getattr(Orig, chain[0]) ]
    for subkey in chain[1:]:
      attrs.append(getattr(attrs[-1], subkey))
    attrs[-1] = Value
    assert len(attrs) == len(chain) + 1

    for idx in reversed(range(len(chain))):
      field, value =  chain[idx], attrs[idx+1]
      # kwarg = dict([[field, value]])
      setattr(attrs[idx], field, value)
    return Orig

# like dataclasses.replace(), but with ability to
# replace fields in nested dataclasses,
# i.e. foo.bar=7 is encoded as foo___bar=7
# Uses ___ as separator, by default
def DCReplace(orig, scopeSep='___', **kwargs):
  if DC.is_dataclass(type(orig)):
    repl = DC.replace(orig)
    for k, v in kwargs.items():
      setDCField(repl, k, v, scopeSep=scopeSep)
    return repl
  return orig

def getDCChangedFields(orig, curr, scopeSep='.'):
  # SPC = " "*(depth+1)
  rtn = OrderedDict()
  fieldNames = OrderedDict(list(map(lambda x: [x.name, x], DC.fields(orig))))
  vals = [ [fn, (getattr(orig, fn), getattr(curr, fn), field)] for fn, field in fieldNames.items() ]
  for fn, (origVal, currVal, field) in vals:
    if DC.is_dataclass(field.type):
      # print(f'{SPC}exploring {fn} {field.type.__name__}')
      subs = getDCChangedFields(origVal, currVal, scopeSep)
      for k, v in subs.items():
        newfield = f'{fn}{scopeSep}{k}'
        rtn[newfield] = v
    else:
      if origVal != currVal:
        #  print(f'{SPC}found {fn} {origVal} {currVal}')
        rtn[fn] = (origVal, currVal)
  return rtn

def overrideConfig(OrigConfig, KK, prompts, printer=None, scopeSep='.'):
  for (theField, theArg, theConverter) in prompts:
    if hasattr(KK, theField):
      Value = theConverter(getattr(KK, theField))
      OrigValue = getDCField(OrigConfig, theField, scopeSep)
      OrigConfig = setDCField(OrigConfig, theField, Value, scopeSep)
      if printer and OrigValue != Value:
        printer(f'overriding {theField}={Value} (was {OrigValue})')
      assert getDCField(OrigConfig, theField) == Value
  return OrigConfig

def convertDictToDataclass(s):
  # used for converting dicts into nested classes
  # i.e. recovering from DC.replace() which replaced a nested dataclass with a dict!
  FN = dict([[f.name, f] for f in DC.fields(type(s))])
  for k, v in FN.items():
    theAttr = getattr(s, k)
    if DC.is_dataclass(v.type) and isinstance(theAttr, dict):
      setattr(s, k, v.type(**theAttr))

def getSchema(obj, depth=1):
  Rtn = []
  if isinstance(obj, dict):
    Rtn = list(obj.keys())
    # print(hprt(' '*(depth), f'starting with {Rtn}'))
    for i in range(len(Rtn)):
      k = Rtn[i]
      if isinstance(obj[k], dict):
        Rtn[i] = (k, getSchema(obj[k], depth+1))
      elif isinstance(obj[k], list):
        Rtn[i] = (k, getSchema(obj[k], depth+1))
  elif isinstance(obj, list):
    Rtn = [ getSchema(i,depth+1) for i in obj ]
  # print(hprt(' '*(depth), f'ending with {Rtn}'))
  return list(filter(lambda x: len(x), Rtn))

def getSchemaLayout(obj, depth=1):
  Rtn = []
  if isinstance(obj, dict):
    Rtn = OrderedDict()
    Keys = list(obj.keys())
    # print(hprt(' '*(depth), f'starting with {Rtn}'))
    for i in range(len(Keys)):
      k = Keys[i]
      if isinstance(obj[k], dict):
        Rtn[k] = getSchemaLayout(obj[k], depth+1)
      elif isinstance(obj[k], list):
        Rtn[k] = getSchemaLayout(obj[k], depth+1)
      else:
        Rtn[k] = type(obj[k]).__name__ if obj[k] is not None else None
        if isinstance(obj[k], bool):
          Rtn[k] = obj[k]
  elif isinstance(obj, list):
    Rtn =  list(filter(lambda x: x is not None, [ getSchemaLayout(i,depth+1) for i in obj ]))
  else:
    Rtn = type(obj).__name__ if obj is not None else None
  # print(hprt(' '*(depth), f'ending with {Rtn}'))
  return Rtn


def splitPath(P):
  rtn = []
  while True:
    head, tail = path.split(P)
    rtn.append(tail)
    if head == '':
      break
    P = head
  return list(filter(lambda x: isinstance(x, str) and len(x) > 0,  reversed(rtn)))


def nonEmptyStrings(ss):
  return list(filter(lambda x: isinstance(x,str) and len(x) > 0,  ss))

def defJSON(x):
  if isinstance(x, set):
    return list(x)
  elif isinstance(x, bytes):
    return x.decode('iso-8859-1')
  elif isinstance(x, DT.date):
    return x.isoformat()
  elif hasAttr(x, 'toJSON'):
    return x.toJSON()
  return x.__dict__

def readLines(fn):
  return filter(lambda x: len(x) > 1, map(lambda x: x.split('#')[0], readOneFile(fn).split("\n")))

def readJSONL(fn):
  return list(map(JSONR.loads, readLines(fn)))

def toJSONStr(x, sort_keys=2, default=defJSON, indent=2, **kwargs):
  return json.dumps(x, default=default,
                    sort_keys=sort_keys, indent=indent, **kwargs)

def toLJSONStr(x, end="\n", sort_keys=False, default=defJSON, indent=None, **kwargs):
  return json.dumps(x, default=default,
                    sort_keys=sort_keys, indent=indent, **kwargs) + end

def fixJSON(jsStr):
  return JSONR.loads(jsStr)


def getAttr(x, f):
  if isinstance(x, dict):
    return x[f]
  else:
    return getattr(x, f)

def setAttr(x, f, v):
  if isinstance(x, dict):
    x[f] = v
  else:
    setattr(x, f, v)
  return x

def qualName(f):
  return f.__qualname__

def numDigits(n, base=10):
  return math.ceil(math.log(n)/math.log(base))

def hasAttr(x, f):
  if isinstance(x, dict):
    return f in x
  else:
    return hasattr(x, f)

def delAttr(x, f):
  if isinstance(x, dict):
    del x[f]
  elif hasAttr(x, f):
    delattr(x, f)
  return x

def getFields(x):
  if isinstance(x, dict):
    return list(x.keys())
  else:
    return vars(x)

def DCField(theType):
  return DC.field(default_factory=theType)

def mergeJSON(x, y, fields=[]):
  if len(fields) == 0:
    fields = y
  for key in fields:
    if hasAttr(y, key):
      setAttr(x, key, getAttr(y, key))
  return x

def extractJSON(x, fields):
  Rtn = {}
  for f in fields:
    if hasAttr(x, f):
      Rtn[f] = getAttr(x,f)
  return Rtn

def delJSON(x, fields=[]):
  for f in fields:
    if hasAttr(x, f):
      delAttr(x, f)
  return x

def writeOneFile(FN, Str):
  mode = "w" if isinstance(Str, str) else "wb"
  # print(f'writing {len(Str)} to {FN}')
  with open(FN, mode) as fff:
    fff.write(Str)

def readOneFile(FN, mode="r"):
  kwargs =  {  "mode" : mode } if mode == "rb" else { "mode" : mode, "encoding" : 'iso-8859-1' }
  with open(FN, **kwargs) as fff:
    return fff.read()


@dataclass
class TensorPrint:
  name:int = 0
  shape:int = 0
  dtype:int = 0
  device:int = 0
  value:int = 0

def isMaybeTensor(x):
  return hasattr(x, 'shape')

def isMaybeNamedParam(x):
  return isinstance(x, tuple) and len(x) == 2 and isinstance(x[0],str) and isMaybeTensor(x[1])

def getTensorSlot(v, slot):
  def wrapRtn(rtn):
    if slot == 'shape' and rtn != '':
      return tuple(rtn)
    return str(rtn)
  rtn = ''
  if isMaybeNamedParam(v):
    rtn = getattr(v[1], slot)
  elif isMaybeTensor(v):
    rtn = getattr(v, slot)
  return wrapRtn(rtn)

def printTensorsAll(*args, which:TensorPrint=TensorPrint(name=1,shape=1,value=1),  **kwargs):
  sh = getShapes(**kwargs)
  picture = [[k] for k in kwargs.keys()]
  _rows = []
  for fn in getDataClassFieldNames(which):
    if getattr(which, fn):
      _rows.append(fn)

  for ci, (k, v) in enumerate(kwargs.items()):
    if which.name:
      # handle named parameter
      # cheap way to test for torch.Tensor without loading it!
      if isMaybeNamedParam(v):
        picture[ci].append(v[0])
      else:
        picture[ci].append(k)
    if which.shape:
      picture[ci].append(getTensorSlot(v, 'shape'))
    if which.dtype:
      picture[ci].append(getTensorSlot(v, 'dtype'))
    if which.device:
      picture[ci].append(getTensorSlot(v, 'device'))
    if which.value:
      if isMaybeNamedParam(v):
        picture[ci].append(hprt(v[1].data))
      elif isMaybeTensor(v):
        picture[ci].append(hprt(v))
      # print(f'did {ri,ci}, {_row} for {k=} {len(picture[ci])=}') 
  rtn = vprt(*args, \
             hprt(vprt(*_rows), \
                  *[vprt(*picture[ci][1:]) for ci in range(len(kwargs))]))
  return rtn

def printTensors(*args, **kwargs):
  return printTensorsAll(*args, **kwargs)

def printTensors0(*args, **kwargs):
  return printTensorsAll(*args, which=TensorPrint(name=1,shape=1,dtype=1,device=1), **kwargs)

def printTensors00(*args, **kwargs):
  return printTensorsAll(*args, which=TensorPrint(shape=1,dtype=1,device=1), **kwargs)

def printTensors1(*args, **kwargs):
  return printTensorsAll(*args, which=TensorPrint(shape=1,dtype=1,device=1,value=1), **kwargs)
