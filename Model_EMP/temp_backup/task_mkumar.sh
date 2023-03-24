#!/bin/bash
#
#SBATCH -p owners --qos mkumar4
#SBATCH -C m640
#SBATCH -J site_name            # name the job fluentP1)
#SBATCH -n 4                                     # number of Nodes to use
#SBATCH --mem=20G                                   # reserve 20G of memory
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
module load math/matlab
matlab -nodisplay -r "main_script; quit"
