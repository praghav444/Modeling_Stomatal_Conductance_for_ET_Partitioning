#!/bin/bash
#
source=$1
site_name=$2
long=$3
lat=$4
Zm=$5
hc=$6
while read source site_name long lat Zm hc; do
    mkdir -p ./$site_name
    cp -r "temp_backup/"* "$site_name/"
    echo $site_name
    sed -i -e "s/tower_height/$Zm/g" $site_name/MyConstants.m
    sed -i -e "s/site_name/$site_name/g" $site_name/main_script.m
    sed -i -e "s/site_name/$site_name/g" $site_name/run_SW_emp_with_opt_params_train.m
    sed -i -e "s/site_name/$site_name/g" $site_name/run_SW_emp_with_opt_params_val.m
done < subset_sites.txt