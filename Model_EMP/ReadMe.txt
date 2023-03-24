## First create directory and related scripts for each site by running the command below:
bash create_dir.sh

## Optimize parameters by running the command below; (Please change the MATLAB executive file path in "run_all_tasks_to_optimize_params.sh" and "prediction.sh" as needed)
bash run_all_tasks_to_optimize_params.sh

## Once we have the optimized parameters, run below to get predictions:
bash prediction.sh