#!/usr/bin/env python3

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

from MyStuff import *

def runOnce(f):
  def wrapper(*args, **kwargs):
    if not hasattr(wrapper, 'has_run') or wrapper.has_run == False:
      wrapper.has_run = True
      return f(*args, **kwargs)
    wrapper.has_run = False
  return wrapper


@runOnce
def setupLoggingHandler():
  global LH1
  global FMT
  LH1 = logging.StreamHandler()
  FMT = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s')
  return LH1, FMT

def startDebug(NAME, LVL=logging.DEBUG, LF=None):
  if NAME is None:
    raise Exception(f'startDebugger needs name!')
  Rtn = logging.getLogger(NAME)
  setupLoggingHandler()
  if Rtn.propagate == True:
    Rtn.propagate = False
    LH1.setFormatter(FMT)
    Rtn.addHandler(LH1)
  Rtn.setLevel(LVL)

  if isinstance(LF, str) and len(lF) > 2:
    LH2 = logging.FileHandler(LF)
    LH2.setFormatter(FMT)
    Rtn.addHandler(LH2)
    Rtn.fileHandler = LH2

  return LogRelay(Rtn)


# class to reduce the number of f-string evaluations if a logging does not happen.
# this uses a modified str.Template to delay the evaluation until the blob passed to
# logger.info call is evaluated to a string.
# is also used to generate "busy status" logs for downstream generation of progress bars
# and such.
# The busyStateStr is simpy a dict of keyword to a tuple of (DONE, TOTAL, WHEN)
# DONE and TOTAL are both int. WHEN is a time.monotonic() stamp
class LogRelay:
  # number of seconds
  BusyFrequency = 1.0
  def __init__(s, lg):
    s.loggers = [ logging.StreamHandler() ]
    if isinstance(lg, LogRelay):
      s.lg = lg.lg
      s.LG = lg
    else:
      assert isinstance(lg, logging.Logger)
      s.lg = lg
      s.LG = s
    s.lg.lastBusy = time.monotonic()
    s.lg.lastBusyLog = 0
    if not hasattr(s.lg, 'busyState'):
      s.lg.busyState = {}
      s.lg.busyChain = []
      s.lg.logStr = ''

  @property
  def lastBusy(s):
    return s.lg.lastBusy

  @property
  def busyChain(s):
    return s.lg.busyChain

  @property
  def busyState(s):
    return s.lg.busyState

  def info(s, fmt, *args, **kwargs):
    s.lg.info(STSP(fmt, *args, **kwargs))
  def debug(s, fmt, *args, **kwargs):
    s.lg.debug(STSP(fmt, *args, **kwargs))
  def warning(s, fmt, *args, **kwargs):
    s.lg.warning(STSP(fmt, *args, **kwargs))
  def error(s, fmt, *args, **kwargs):
    s.lg.error(STSP(fmt, *args, **kwargs))
  def critical(s, fmt, *args, **kwargs):
    s.lg.critical(STSP(fmt, *args, **kwargs))
  def exception(s, fmt, *args, **kwargs):
    s.lg.exception(STSP(fmt, *args, **kwargs))
  def addFileLogger(s, fn, maxBytes=100000, backupCount=5):
    Rtn = len(s.handlers)
    s.handlers.append(looging.RotatingFileHandler(fn, mode, maxbytes, backupCount))
    s.lg.addHandler(s.handlers[-1])
    return Rtn

  # idx=0 is for the default StreamHandler
  def removeLogger(idx = -1):
    s.lg.removeHandler(s.handlers[idx])

  def addBusyChain(s, *args):
    for LG in args:
      assert isinstance(LG, LogRelay)
      s.lg.busyChain.append(LG)

  def setBusyNow(s, now):
    # logging here is not helpful
    # if now - s.lastBusy > s.BusyFrequency:
    #   s.lg.info(s.busyStateStr(now))
    s.lg.lastBusy = now

  @property
  def lastBusy(s):
    return s.lg.lastBusy

  @property
  def lastBusyLog(s):
    return s.lg.lastBusyLog

  # get the last recoede
  @property
  def logStr(s):
    return s.lg.logStr

  def setBusyLogStr(s, now, ss):
    s.lg.lastBusyLog = now
    s.lg.logStr = ss
    return ss

  def busyStart(s, Kind, N=1, now=None):
    # - adds N to total number of Kind {e.g. Files, Bytes, Tasks)
    if now is None:
      now = time.monotonic()
    s.setBusyNow(now)
    Entry = s.busyState.get(Kind, [0, 0, now])
    Entry[1:] = [ Entry[1] + N, now ]
    s.lg.busyState[Kind] = Entry
    return Entry

  def setNotBusyAll(s, now=None):
    if now is None:
      now = time.monotonic()
    for c in s.busyChain:
      c.lg.busyState = {}
    s.lg.busyState = {}

  def clearIfNotBusy(s, now = None):
    if now is None:
      now = time.monotonic()
    for Kind in s.lg.busyState.keys():
      Entry = s.lg.busyState.get(Kind, [0,0, now])
      if Entry[0] and Entry[0] == Entry[1]:
        Entry = [0, 0, now]
        s.lg.busyState[Kind] = Entry

    for c in s.busyChain:
      c.clearIfNotBusy(now)

  def busyFinish(s, Kind, N=1, now=None):
    # adds N to the number completed of Kind (e.g. Files, Bytes, Tasks)
    if now is None:
      now = time.monotonic()
    s.setBusyNow(now)
    Entry = s.lg.busyState.get(Kind, [0,0, now])
    # if  Entry[0] + N >= Entry[1]:
    #   Entry = [0, 0, now]
    # else:
    Entry = [Entry[0]+N, Entry[1], now]
    s.busyState[Kind] = Entry
    return Entry

  @property
  def getBusyState(s):
    rtn = list(s.busyState.items())
    for c in s.busyChain:
      rtn.extend(list(c.busyState.items()))
    Rtn = dict()
    # merge repeating keys
    for k, (done,total,when) in rtn:
      Entry = Rtn.get(k, [0, 0, 0])
      Entry[0] += done
      Entry[1] += total
      Entry[2] = max(when, Entry[2])
      Rtn[k] = Entry
    return Rtn

  @property
  def getLastBusy(s):
    lastBusy = s.lastBusy
    return max(lastBusy, lastBusy, *[ c.lastBusy for c in s.busyChain])

  def busyStateStr(s, now=None):
    # if the last time the busyStateStr() was called
    # is later than the last event, then we can just return it
    lastBusy = s.getLastBusy
    LastStr = ''
    if s.lastBusyLog > lastBusy:
      if len(s.logStr):
        if now is None:
          now = time.monotonic()
        Last = now - lastBusy
        LastStr = '' if Last < 0.001 else f' {Last:0.3f} seconds ago'
      return s.logStr + LastStr
    if now is None:
      now = time.monotonic()
    # otherwise generate the logStr
    activity = s.getBusyState
    Res = []
    Last = now
    LogStr = ''
    LastStr = ''
    if len(activity):
      Res = [ f'{kind} {done}/{total}' for kind, (done,total, when) in activity.items() ]
      Times = [ now - when for kind, (done,total,when) in activity.items() ]
      assert len(Times) == len(Res)
      Last = min(now, *Times)
    if len(Res):
      LastStr = '' if Last < 0.001 else f' {Last:0.3f} seconds ago'
      LogStr = " ".join(Res)
    if len(LogStr):
      return s.setBusyLogStr(now, LogStr) + LastStr
    return LogStr

def myInject(fromClass, toClass, methodNames=None, Force=False):
  if methodNames is None or not(isinstance(methodNames, (list,set))):
    methodNames = set(list(filter(lambda x: not x.startswith('__'), dir(fromClass))))
  for name in methodNames:
    if not hasattr(fromClass, name) or (not Force and hasattr(toClass, name)):
      continue
    the = getattr(fromClass, name)
    setattr(toClass, name, the)

def injectLogRelay(toClass, instances=[], lgName = '', lgLevel=logging.DEBUG, Force=False):
  if not hasattr(toClass, 'setBusyNow'):
    # print('injecting class')
    myInject(LogRelay, toClass, Force=Force)
  lgName = toClass.__qualname__ if len(lgName) < 1 else lgName
  # print(f'{lgName=}')
  if len(lgName):
    LG = startDebug(lgName, lgLevel)
    setattr(toClass, 'LG', LG)
    if not isinstance(instances, (set,list)):
      instances = [instances]
    assert len(instances)
    for ii in instances:
      if not isinstance(ii, toClass):
        # print('skip')
        continue
      # print('injecting instance')
      myInject(LG, ii, ['LG', 'lg'], Force=Force)
    return LG

def setupLogRelay(self, lgName='', lgLevel=logging.DEBUG, Force=False):
  toClass = type(self)
  injectLogRelay(toClass, self, lgName, lgLevel, Force)


if __name__ == "__main__" and sys.argv[0].endswith("LogRelay.py"):
  LG1 = startDebug(__name__)
  LG2 = startDebug(f'{__name__}.submodule')
  assert LG1.lg != LG2.lg
  class Outer(LogRelay):
    def __init__(s, bar):
      LogRelay.__init__(s, LG1)
      s.addBusyChain(bar.LG)

  class Inner(LogRelay):
    def __init__(s):
      LogRelay.__init__(s, LG2)

  now = 17
  B = Inner()
  A = Outer(B)
  A.setBusyNow(now)
  B.setBusyNow(now)
  A.busyStart('Task1', now=now)
  A.busyStart('Task2', now=now+1)
  B.busyStart('Task2', now=now+2)

  assert A.busyState == {'Task1': [0, 1, 17], 'Task2': [0, 1, 18]}
  assert B.busyState == { 'Task2': [0, 1, 19]}
  # print(A.getBusyState)
  assert A.getBusyState == {'Task1': [0, 1, 17], 'Task2':[0, 2, 19]}
  assert A.lastBusyLog < A.lastBusy
  assert A.busyStateStr(now=now+3) == 'Task1 0/1 Task2 0/2 1.000 seconds ago'

  assert A.lg.logStr == 'Task1 0/1 Task2 0/2'
  # print(A.LG.lg.busyState)
  # print( A.busyStateStr(now=now+4))
  assert A.busyStateStr(now=now+4) == 'Task1 0/1 Task2 0/2 2.000 seconds ago'
  assert A.lastBusy == now+1
  assert B.lastBusy == now+2
  assert A.getLastBusy == now+2
