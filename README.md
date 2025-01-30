# SimpleLlama3.X

  This repo is mainly for educational purposes.
  I had lots of questions about Llama models (and AI/LLMs, in general) so I made this repo with the Jupyter Notebooks inside, 
  to help me learn the finer points of how LLMs actually work.
  As I learn and improve, I'll periodically update this repo with additional materials.

## Take a look at <a href="./REQUIREMENTS.md">REQUIREMENTS.md</a>
  - It contains the instructions on how to set things up for jupyter
    (You can skip this if you already have `jupyter` in your PATH)

## Code Descriptions:
  - `subroutines.py` 
    - This is just a random collection of my own utility functions I found to be useful, including pretty printing of `torch.Tensor`
    - It also has some helpful routines for dealing with python3 `@dataclass` classes, such as nested dataclass accessors (get AND set), and turning a `@dataclass` fields to command line switches for modifying the @dataclass instance at run time with command line switches.
    - ```
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
      ```
      results in
      ```
      about to patch sys.version_info(major=3, minor=10, micro=14, releaselevel='final', serial=0) string.Template
      2024-10-29 16:11:56,893:INFO:MyConfigUser: overriding foo=9 (was 0)
      2024-10-29 16:11:56,893:INFO:MyConfigUser: overriding somefoo.val=7.0 (was 8.0)
      {'foo': 9, 'somefoo': {'val': 7.0}}
      2024-10-29 16:11:56,893:INFO:MyConfigUser: Now I am enabled for logging! w00T!
      ```
      So in three lines of code, you can reconfigure your `@dataclass` instance at run time (using command line arguments!) without much hassles.
      
  - `LogRelay.py`
    - I never liked python3's f-string syntax due to the fact that it slowed things down, even when a log message was not actually emitted. (because it is interpolated whether or not the actual log message happens!) This is an attempt to inject logger.logger functionality to arbitrary classes, while delaying the message string interpolation until actual output time, regardless of what kind of inheritence is used (to super(), or not to super(), that is the question! :-). I hacked up a near-foolproof way for any class to get self.info, self.debug, self.error, etc... capabilities that can be done at any time (most conveniently, during its `__init__()` but it's not limited to that.
    - It also has the ability to keep track of "busy state". This bit of code is not used here, but...)

# TinyTorch

  This is a small (near-drop-in) replacement for pytorch.Tensor class
  that is designed for easy study. I started to write this, simply to
  understand how some of pytorch's capabilities are actually
  implemented. The TinyTorch.Tensor class support `backward()` just like `pytorch.Tensor`

  This package is intended as a demonstration of some of the cool internal
  details of pytorch, and how some of these cool features are implemented.

  For reducing code size, I split up the data access portions of Tensor
  into TinyArray. TinyArray implements most of the data layout and access logic
  (i.e. similar to (but way simpler than))  numpy arrays and `pytorch.Tensor`

  TinyTensor implements some of torch.Tensor access logic. In almost all cases,
  I took pains to showcase the simplest implementation that mimimcs
  pytorch's behavior. Many obvious optimizations were avoided for the sake
  of code simplicity.

  While this package *CAN* be used to design and test neural networks,
  it is not meant to do so. (Just use pytorch!) This package is purely
  for education purposes, meant to teach how some of the clever techniques
  employed by pytorch could be implemented.

## `tinytorch/MyStuff.py`
  Few utlities, including a DebugHelper class which is designed to showcase
  a pattern for how to implement a helper class that can be used to keep track
  of which flags (for debug out) are enabled, etc...

  ```
  DebugPush(...)
    pushes a set of flags to activate debugging facility.
    flags are typically things like
    CLASS   - i.e. debug only this particular class
    CLASS.METHODNAME - debug this particular method in the class CLASS
    INSTANCE -  (debug only this instance)
    INSTANCE.METHOD - (debug only this instance's method)
    anything else - you can custom craft Debug flags. i.e. strings, sets, dicts, ...


  DebugPop()
    pops the topmost set of flags. Does not empty the stack

   ```
## `tinytorch/tensorhelpers.py`
  Helper routines for pretty printing.
  ```
  vprt(*args, **kwargs) - vertically align each argument
  hprt(*args, **kwargs) - horizontally align each arguement
  ```
  `hprt` and `vprt` can be combined to present an "ascii-art" picture
  of its arguments.
  For example, I find it useful to horizontally paste the name of a Tensor,
  then its shape, followed by its actual elements.

## `tinytorch/view.py`
  Utility to draw DOT diagrams of expression trees in both Value and Tensor

# Credits! (I stood on the shoulders of giants!)
  - Umar Jamil's [pytorch-llama](https://github.com/hkproj/pytorch-llama) 
  - Andrej Karpathy's  [micrograd](https://github.com/karpathy/micrograd)
  - [Meta Llama Official Repository](https://github.com/meta-llama/llama-models)
  - [Official Meta Llama Model Download link](https://www.llama.com/llama-downloads/)