#!/usr/bin/bash
pushd . > /dev/null
SCRIPT_PATH="${BASH_SOURCE[0]}";
if ([ -h "${SCRIPT_PATH}" ]) then
  while([ -h "${SCRIPT_PATH}" ]) do cd `dirname "$SCRIPT_PATH"`; SCRIPT_PATH=`readlink "${SCRIPT_PATH}"`; done
fi
cd `dirname ${SCRIPT_PATH}` > /dev/null
SCRIPT_PATH=`pwd`;

wget ftp://ftp.cscs.ch/in/put/abc/cosmo/fuo/physics_standalone/microph/data.tar.gz
test -f data.tar.gz || exit 1
tar -xvf data.tar.gz --directory ${SCRIPT_PATH}/data_microph || exit 1
/bin/rm -f data.tar.gz 2>/dev/null