#  Requirements
 ## You'll probably want to install NVIDIA drivers on your machine (if you have an NVIDIA GPU)

   - (For Ubuntu) <https://ubuntu.com/server/docs/nvidia-drivers-installation>
   - (Directly from NVIDIA) <https://docs.nvidia.com/cuda/cuda-installation-guide-linux>
   - Go here to download the nvidia drivers <https://developer.nvidia.com/cuda-downloads>
   - This example assumes zsh/bash in LINUX or a linux-like environment.
   - NOTE! `.` (SINGLE PERIOD) means to load the script and execute it IN THE RUNNING SHELL
   - NOTE!  ``$'' is the typical unix shell prompt!

 ## python3 (>=3.11 preferred)
 ## You can either use miniconda or directly use pip.
 ### Using Miniconda makes dealing with different versions of python easier.
 ## Setup your python virtual environment using pip
   - (assuming your version is python3.11)
   - This example assumes zsh/bash in LINUX or a linux-like environment.
   - If you have to deal with multiple versions of python3, consider using conda to boostrap the venv.
   - NOTE! `.` (SINGLE PERIOD) means to load the script and execute it IN THE RUNNING SHELL
   - NOTE!  ``$'' is the typical unix shell prompt!
   - `$ mkdir -p ~/VEnvs`
   - `$ python3 -m venv ~/VEnvs/torch-311`
   - NOTE! the directory `~/VEnvs/torch-311` does not matter. Just pick one that you'll use for this project.
   - The above python call will create the `~/VEnvs/torch-311` and populate it for installing new python packages locally within it without disturbing the global python environment
   - There will be several `activate` scripts that can be used to modify the running shell's environment to point to the module directory for all of the locally installed python packages.
   - NOTE! pick the right "activate"  for your shell!
   - `$ . ~/VEnvs/torch-311/bin/activate`
   - Now your prompt should change, like below.
   - `(torch-311) $ python3 -m pip install --upgrade pip`

 ## Install and use miniconda to manage different "pip" venvs without having to remember where they are.
   - <https://docs.anaconda.com/free/miniconda/>
   - <https://www.anaconda.com/blog/understanding-conda-and-pip>
   - The advantage of this method is that you can now maintain multiple venvs. You only need to remember where you installed `miniconda`
   - Each Miniconda environment can have a different version of python as well as different versions of packages.
   - Pip does not handle multiple python versions by itself.
   - Assuming you installed miniconda in `~/miniconda`
   - `$ . ~/miniconda3/bin/activate`
   - `(base) $ ` By default `conda` starts off with a venv called `base` with the latest and greatest python3.
   - You probably should not pollute the `base` env with lots of stuff.
   - A good choice is `(base) $ conda create -n tiny`
   - `(base) $ conda activate tiny`
   - `(tiny) $ `
   - In general `pip install PACKAGENAME` unless the package specifically requires `conda` to install.
   - Goto <https://pytorch.org/get-started/locally/> and click on `pip` for the package to see the current way to install the latest pytorch.
       - This command is currently `pip install pytorch torchvision torchaudio`
           - installs the latest pytorch and nvidia support into the current environment.
   - `(tiny) $ pip install graphviz numpy scipy simpy jupyter` to install some additional packages.

 ### Managing VENvs with MiniCONDA
   - Here are some more examples of what you can do with conda
   - `(base) $ conda create -n v310 python=3.10`
       - This will create a new venv called `v310` and start it off with python 3.10!
       - You can still MANUALLY create ADDITIONAL Venvs using `python -m venv NEWDIRNAME` but you really don't need to any more.
   - `(base) $ conda activate v310`
     - You no longer need to remember where you placed the venv. Conda does that for you.
   - `(v310) $ ` Now using python3.10, you can use `pip` to install packages.
   - In general, use `pip` to install packages unless it requires conda to install.
   - `(v310) $ conda install pytorch==2.2.2 pytorch-cuda=12.1 cudatoolkit -c pytorch -c nvidia`
       This will install a specific version of pytorch into the current conda env `v310`
   - `(v310) $ conda deactivate` to leave the conda environment


 ## Install pytorch
   - You can get by with the CPU version of pytorch if you are having difficulties getting your GPU drivers to work
   - (Pytorch) <https://pytorch.org/get-started/locally/>

 ## Now add additional packages using pip
   - `$ pip install numpy simpy scipy matplotlib graphviz jupyter`
   - `$ pip list` to see the list of installed packages int the current venv
   - `$ pip show torch` to see details about the `torch` package
   -
 ## leave the pip venv when you're done
   - `(torch-311) $ deactivate`
   - `$ `

# Running Jupyter.
  - The below works in zsh.
    The below presumes that "conda" is available in your shell.
    You put it in your .zshrc
    ```
    jstart() {
      local MYDIR=$1
      local MYENV=$2
      local PORT=$3
      if [ -z "${PORT}" ]; then
        PORT=8888
      fi
      echo $PORT

      if ! [ -x "${CONDA_EXE}" ]; then
        echo you need to source a CONDA environment
        return
      fi
      if ! [ -n "${MYDIR}" -a -d "${MYDIR}" ]; then
        echo "SUPPLY A DIRECTORY was '${start}'";
        return
      fi
      if  [ -z "${MYENV}" ]; then
        echo SUPPLY A CONDA ENV NAME
        return
      fi

      conda activate $MYENV
      (cd  $MYDIR; jupyter notebook --no-browser --port=$PORT)
    }
    ```
    or Take a look at <a href="./jupyter.sh">jupyter.sh</a> in this repo
    and source it using `. jupyter.sh`
    Then you can `jstart DIR VENV`
    and open your browser to the location it specifies to start Jupyter
