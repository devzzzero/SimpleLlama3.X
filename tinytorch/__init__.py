PRELOADED_MODULES = set()

import sys
import os
from os import path
execdir, execname = path.split(sys.argv[0])
cwd = os.getcwd()
allCwds = [ cwd, execdir ]
Found = False
for ncwd in allCwds:
  if path.isdir(ncwd) and path.isfile(path.join(ncwd, 'TinyArray.py')):
    sys.path.append(ncwd)
    Found = True
    break
  ncwd1 = path.join(ncwd, 'tinytorch')
  if path.isdir(ncwd1) and path.isfile(path.join(ncwd1, 'TinyArray.py')):
    sys.path.append(ncwd1)
    Found = True
    break
if not Found:
    raise Exception(f'{execdir=} {cwd=} is not a valid place to find {sys.argv[0]}')

# print (sys.path)
from MyStuff import *
from TinyArray import *
from TinyTensor import *
from tensorhelpers import *
from view import *

