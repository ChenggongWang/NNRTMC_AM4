#!/bin/bash

###################################################################
#Script Name : run_fregrid.sh
#Description : 
#Args        : 
#Author      : Chenggong Wang 
#Email       : c.wang@princeton.edu  
###################################################################

# exit when any command fails
set -e
# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT
# command starts here

echo $SHELL
source /tigress/cw55/stellar/FRE-NCtools/build/env.stellar.sh
export fregrid=/tigress/cw55/stellar/FRE-NCtools/build/bin/fregrid

$fregrid --input_mosaic C96_mosaic.nc \
	--input_file $1 \
	--interp_method conserve_order1 \
	--remap_file fregrid_remap_file \
	--nlon 288 --nlat 180\
	--scalar_field $2

