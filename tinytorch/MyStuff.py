#!/usr/bin/python3
from os import path
import sys
import re
import copy
import string
from functools import reduce
from os import path
import ntpath
import logging
import time

if sys.version_info.major == 3 and sys.version_info.minor < 11:
  from types import MethodType
  def get_identifiers(self):
    ids = []
    for mo in self.pattern.finditer(self.template):
      named = mo.group('named') or mo.group('braced')
      if named is not None and named not in ids:
        # add a named group only the first time it appears
        ids.append(named)
      elif (named is None
        and mo.group('invalid') is None
        and mo.group('escaped') is None):
        # If all the groups are None, there must be
        # another group we're not expecting
        raise ValueError('Unrecognized named group in pattern',
          self.pattern)
    return ids
  if not hasattr(string.Template, 'get_identifiers'):
    sys.stderr.write(f'about to patch {sys.version_info} string.Template\n')
    setattr(string.Template, 'get_identifiers', get_identifiers)

# delayed parsing of the format until its cast into a str
class STSP:
  def __init__(s, fmt, *args, **kwargs):
    s.fmt = fmt
    s.args = args
    s.kwargs = kwargs
    # print(f'STS initialized with {fmt} {len(s.args)} {len(s.kwargs)}')

  @staticmethod
  def eval(fm, *args, **kwargs):
    return parseTemplateStr(s.fmt, *s.args, **s.kwargs)

  @staticmethod
  def eval0(fm, *args, **kwargs):
    return parseTemplateStr(s.fmt, *s.args, **s.kwargs)[0]

  def __str__(s):
    # print('parsing now')
    return parseTemplateStr(s.fmt, *s.args, **s.kwargs)[0]

# An attempt to avoid evaluating the format string before
# deciding to log something
# not perfect. Any modifications to arguments are applied
# before
# also



def runOnce(f):
  def wrapper(*args, **kwargs):
    if not hasattr(wrapper, 'has_run') or wrapper.has_run == False:
      wrapper.has_run = True
      return f(*args, **kwargs)
    wrapper.has_run = False
  return wrapper



def getWinPath(fn):
  """ parses the path and returns a proper Windows path
  """
  DRV, FNC  = ntpath.splitdrive(fn)
  FNA = []
  while FNC:
    (hh,tt) = path.split(FNC)
    if (tt):
      FNA.append(tt)
      if hh != "/":
        FNC = hh
      else:
        FNA.append("\\")
        break
  FNA.reverse()
  #print(DRV, FNA)
  WinFNC = ntpath.join(DRV, *FNA)
  return WinFNC




###
def LOG2(x):
  from math import ceil,log
  return int(ceil(log(x)/log(2)))


def CP(a):
  return copy.deepcopy(a)

def listInstanceOf(i, theType=int, op=lambda x: True):
  # if len(i) < 1:
  #   return True
  #   raise ValueError(f'{i=} is not a valid list')
  # if len(i) == 1 and isinstance(i[0], (list, tuple)) and op(x):
  #   i = i[0]
  K = ADD(*[ isinstance(x, theType) and op(x) for x in i], 0)
  # print(f'{i=} {theType=} {op=} {len(i)} ==? {K=}')
  return len(i) == K




def RXC(*args, **kwargs):
  return re.compile(*args, **kwargs)

def RXS(rx, *args, **kwargs):
  return re.search(rx, *args, **kwargs)


def RXM(rx, *args, **kwargs):
  return re.match(rx, *args, **kwargs)


def ST(f):
  return string.Template(f)
def TSP(f, *args, **kwargs):
  return f.substitute(*args, **kwargs)

def STSPX(f, *args, **kwargs):
  return parseTemplateStr(f, *args, **kwargs)

def STSP0(f, *args, **kwargs):
  return parseTemplateStr(f, *args, **kwargs)[0]



def parseTemplateStr(fmt, *args, **kwargs):
  pos0 = fmt.find('$')
  missingKw = []
  UsedLen = 0
  R = fmt
  if pos0 >= 0:
    F0 = string.Template(fmt)
    idl = F0.get_identifiers()
    if IsDebugging(parseTemplateStr):
      print(f'{fmt} has {idl}')
    for v in idl:
      if v not in kwargs:
        missingKw.append(v)
    # then all keywords are in kwargs
    UsedLen = 0
    if len(missingKw) and len(args):
      UsedLen = min(len(missingKw), len(args))
      for kw, val in zip(missingKw, args):
        kwargs[kw] = val
    R = F0.safe_substitute(**kwargs)
  return R, UsedLen, missingKw


def prt(*args, **kwargs):
  def SPC(c):
    return "  "*c
  L = 1
  if "l" in kwargs:
    L = kwargs["l"]
  if L > 1:
    print()
  for arg in args:
    if isinstance(arg, list):
      print(SPC(L),"[ ")
      for i,v in enumerate(arg):
        print(SPC(L+1), i, ":", v, ", ")
      print(SPC(L),"]")
    elif isinstance(arg, tuple):
      print(SPC(L),"( ")
      for i,v in enumerate(arg):
        print(v, ", ", end=' ')
      print(")")
    elif isinstance(arg,set):
      print(SPC(L), "set(")
      for i,v in enumerate(arg):
        print(SPC(L+1),v, ", ")
      print(SPC(L), ")")      
    elif isinstance(arg, dict):
      print(SPC(L), "{ ")
      for k, v in arg.items():
        print(SPC(L+1), k, ": ", end=' ')
        prt(v, l=L+2), ", "
      print(SPC(L), "}")
    else:
      print(SPC(L), arg)

def InitDbg(s, printer=None):
  DebugHelper.__init__(s, printer)


def OR(a, b, *c):
  return reduce(lambda x, y: x | y, c, a | b)

def ADD(a, b, *c):
  return reduce(lambda x, y: x + y, c, a + b)


def AND(a, b, *c):
  return reduce(lambda x,y: x & y, c, a & b)

def MUL(a, b, *c):
  return reduce(lambda x,y: x * y, c, a * b)


def DebugFlags(*args):
  rtn = set([])
  for arg in args:
    if isinstance(arg,set):
      rtn = rtn | arg
    elif isinstance(arg, list) or isinstance(arg, tuple):
      rtn = rtn | reduce(OR, [ DebugFlags(arg1) for arg1 in arg ])
    elif isinstance(arg, int) or isinstance(arg, str) or isinstance(arg, type):
      rtn = rtn | set([arg])
    else:
      rtn = rtn | set([arg])
  return rtn


    

def DebugPrintWrapper(thePrinter=None, *args,  **kwargs):
  if thePrinter == print or thePrinter == None:
    ## filter out **kwargs to only have what default print wants
    ## see if args[0] (the format) has '$'
    R, FromArgs, MissingKw = STSP0(*args, **kwargs)
    provided = set.intersection(DebugHelper.__def_print_kwargs__, set(kwargs.keys()))
    nkw = dict([ (k, kwargs[v])  for k in provided ])
    print(R, *args[1+FromArgs:], **nkw)
  else:
    # else just pass them through
    thePrinter(*args, **kwargs)

def DbgPrt(f, *args, **kwargs):
  if IsDebugging(f):
    thePrinter=findDebugPrinter()
    DebugPrintWrapper(thePrinter, *args, **kwargs)

def PrtDbg(*args, **kwargs):
  thePrinter=findDebugPrinter()
  DebugPrintWrapper(thePrinter, *args, **kwargs)

def setDefaultPrinter(printer):
  DebugHelper.__def_printer__ = printer

def findDebugPrinter(s = None, offset=-1):
  if s and s.__printer__:
    return s.__printer__
  p = CurrDebugPrinter(offset)
  if p:
    return p
  if DebugHelper.__def_printer__:
    return DebugHelper.__def_printer__
  return print


class DebugHelper(object):
  """ Keeps track of debugging flags (and an optional printer to use) """
  """ Also maintains a stack of debug flag sets (and optional printer to use)"""
  """ The local class (i.e. classes that inherit from DebugHelper) """
  """ can also have its own printer function """
  """ Do not used f-strings as arguments!"""
  """ That will result in odd behavior even when you are not debugging!"""
  """ Use STSP(fmt, ..., kw=...)"""
  """ STSP substitutes unnamed args with the named args, in appearance order """
  """ STSP can also take actual keyword arguments """
  __debug_on__ = [[set([]),None]]
  __def_print_kwargs__ = set(['sep','end','file','flush'])
  __def_printer__ = None
  @staticmethod
  def __init__(s, printer=None):
    s.__debug_class__ = set([s.__class__])
    s.__printer__ = printer
  def setPrinter(s, printer):
    s.__printer__ = printer
  def findPrinter(s, offset=-1):
    return findDebugPrinter(s, offset=offset)

  def printer(s, *args, **kwargs):
    thePrinter = s.findPrinter()
    DebugPrintWrapper(thePrinter, *args, **kwargs)

  @property
  def isDebugging(s):
    flags = DebugFlags(s.__debug_class__)
    return (CurrDebugFlags() & flags) == flags

  def Debugging(s, *f):
    flags = DebugFlags(*f) | s.__debug_class__
    return (CurrDebugFlags() & flags)==flags

  def DBG2(s, f, *args, **kwargs):
    if s.Debugging(f):
      s.printer(*args, **kwargs)
  def DBG3(s, f, cond, *args, **kwargs):
    if s.Debugging(f) and cond:
      s.printer(*args, **kwargs)
  def DBG(s, *args, **kwargs):
    if s.isDebugging:
      s.printer(*args, **kwargs)

def IsDebugging(*f, idx=-1):
  flags = DebugFlags(*f)
  return (CurrDebugFlags(idx) & flags)==flags


def CurrDebugFlagEntry(id=-1):
  return DebugHelper.__debug_on__[id]
def CurrDebugFlags(id=-1):
  return DebugHelper.__debug_on__[id][0]
def CurrDebugPrinter(id=-1):
  return DebugHelper.__debug_on__[id][1]

def DebugNow(*args, **kwargs):
  f = DebugFlags(*args, **kwargs)
  if DebugHelper in CurrDebugFlags():
    print("setting debug flags to", f)
  DebugHelper.__debug_on__[-1][0] = f


## push a debuggings state
def DebugPush(*args, printer=None):
  f = DebugFlags(*args)
  if DebugHelper in CurrDebugFlags():
    print("setting debug flags to", f)
  DebugHelper.__debug_on__.append([f, printer])

# does not empty out the DebugFlag stack!
def DebugPop():
  if len(DebugHelper.__debug_on__) > 1:
    rtn = DebugHelper.__debug_on__[-1]
    DebugHelper.__debug_on__ = DebugHelper.__debug_on__[:-1]
    return rtn
  return DebugHelper.__debug_on__[-1]

if __name__ == "__main__" and sys.argv[0].endswith("MyStuff.py"):
  a=RXC("""(a|b)c*""")
  print(a)
  b=RXS(a, """fooaaaaa""")
  print(b)
  print(b.groups(0), b.start(0), b.end(0))
          

  x = { 1 : 2, 2: ["asass",4,5,6] }
  prt(x)
  y = [ 1,2,3 ]
  prt(y)

  class X(DebugHelper):
    def __init__(s):
      InitDbg(s)
    
  print(isinstance(int, type), isinstance(X, type))
  print(STSP("barf=$barf, bag=%bag",barf="aa",bag="bb"))
  xx = X()
  print(xx.__debug_on__)
  print("fooo", DebugFlags("aaa",1,2,3))
  prt(xx.__debug_class__)
  DebugNow(DebugHelper, X)
  prt(xx.__debug_on__)
  print(xx.Debugging(), CurrDebugFlags())
  DebugPush(X)
  print(xx.Debugging(), CurrDebugFlags())
  DebugPush(1)
  print(xx.Debugging(), CurrDebugFlags())
  # DebugPush([1,2,3])
  # print(xx.Debugging(), CurrDebugFlags())                     
  # print(DebugPop(), xx.Debugging(), CurrDebugFlags())
  # print(DebugPop(), xx.Debugging(), CurrDebugFlags())
  # print(DebugPop(), xx.Debugging(), CurrDebugFlags())
  # print(DebugPop(), xx.Debugging(), CurrDebugFlags())
  # print(LOG2(255))



  # print(ord("a"))
  assert (listInstanceOf([[0,1],[2,3,4]], theType=(tuple,list)))
  assert (listInstanceOf([[0,1],[2,3,4]], theType=list))
  assert not (listInstanceOf([[0,1],[2,3,4]], theType=tuple))
  assert not (listInstanceOf([[0,1],[2,3,4]], theType=tuple, op=lambda x: len(x) == 2))
  assert (listInstanceOf([[0,1],[2,3]], theType=list, op=lambda x: len(x) == 2))
  assert (listInstanceOf([[0,1],[2,3]], theType=(list,tuple), op=lambda x: len(x) == 2))
  assert not listInstanceOf([[0,1],[2,3,4]], theType=(list,tuple), op=lambda x: len(x) == 2)

  # DebugPush(parseTemplateStr)
  assert parseTemplateStr('hello${world}123${world} $world yes', 1) == ('hello11231 1 yes',1, ['world'])
  assert parseTemplateStr('hello${world}123${world} $world yes', world='X') == ('helloX123X X yes',0, [])
  assert parseTemplateStr('hello${world1}123${world2} $world yes', world='X') == ('hello${world1}123${world2} X yes', 0, ['world1', 'world2'])
  assert parseTemplateStr('hello${world1}123${world2} $world yes', ' UP ', world='X') == ('hello UP 123${world2} X yes',1, ['world1', 'world2'])
  assert parseTemplateStr('hello${world1}123${world2} $world yes', ' UP ', ' DOWN ', world='X') == ('hello UP 123 DOWN  X yes', 2, ['world1', 'world2'])
  assert parseTemplateStr('hello${world1}123${world2} $world yes',  ' DOWN ', world1=' UP ',world='X') == ('hello UP 123 DOWN  X yes', 1, ['world2'])
  assert parseTemplateStr('hello${world1}123${world2} $world yes',  world2=' DOWN ', world1=' UP ',world='X') == ('hello UP 123 DOWN  X yes', 0, [])

  DebugPush('hello')
  DbgPrt('hello', 'none substituted: $arg $foo $bar')
  DbgPrt('hello', '7 substituted: $arg $foo $bar', 7)
  DbgPrt('hello', '7 substituted: $arg $foo $bar', 7, 8, 9)
  
