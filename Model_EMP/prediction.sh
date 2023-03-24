#!/bin/bash
#
source=$1
site_name=$2
long=$3
lat=$4
Zm=$5
hc=$6
while read source site_name long lat Zm hc; do
    cd $site_name
    "/Applications/MATLAB_R2022a.app/bin/matlab"  -nodisplay -nodesktop -r "run('run_SW_emp_with_opt_params_train.m'); exit;"
    "/Applications/MATLAB_R2022a.app/bin/matlab"  -nodisplay -nodesktop -r "run('run_SW_emp_with_opt_params_val.m'); exit;"
    cd ..
done < subset_sites.txt