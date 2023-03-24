#!/bin/bash
#
#SBATCH --qos=medium
#SBATCH --time=100:00:00
#SBATCH -n 1                                     # number of Nodes to use
#SBATCH -J AT-Neu            # name the job fluentP1)
#SBATCH --mem=8G                                   # reserve 20G of memory
#SBATCH -o slurm-results.out                       # standard output file
#SBATCH -e slurm-error.err                         # standard error file
#SBATCH --mail-type=END,FAIL                       # notifications for job completion/failure
#SBATCH --mail-user=ppushpendra@crimson.ua.edu        # send to address
#SBATCH --mail-type=ALL
srun hostname -s > hosts.$SLURM_JOB_ID
#
echo "I ran on:"
cd $SLURM_SUBMIT_DIR
echo $SLURM_NODELIST
#
module load matlab
matlab -nodisplay -r "main_script; quit"
