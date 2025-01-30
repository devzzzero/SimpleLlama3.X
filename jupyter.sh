#!/bin/zsh

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
