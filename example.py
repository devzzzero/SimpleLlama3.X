import dataclasses
import argparse

from subroutines import *
import dataclasses as DC
import argparse
from subroutines import *
from LogRelay import *

@dataclass
class SomeFoo:
  val:float = 8.0

@dataclass
class SomeDataClass:
   foo:int = 0
   somefoo:SomeFoo = DCField(SomeFoo) # dataclasses.field(default_factory=SomeFoo)

class MyConfigUser(SomeDataClass):
  def reconfigure(self, *argv):
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, allow_abbrev=True)
    prompts = buildCmdLineParserFrom(SomeDataClass)
    for (theField, theArg, theConverter) in prompts:
      parser.add_argument(theArg, type=theConverter)
    KK = parser.parse_args(*argv)
    self.prompts, self.KK = prompts, KK
    self = overrideConfig(self, KK, prompts, printer=self.info)
    return self

cc = MyConfigUser()
setupLogRelay(cc)
cc.reconfigure(splitByWhiteSpace('''--somefoo.val=7
  --foo=9
   '''))
print(DC.asdict(cc))
cc.info('Now I am enabled for logging! w00T!')
