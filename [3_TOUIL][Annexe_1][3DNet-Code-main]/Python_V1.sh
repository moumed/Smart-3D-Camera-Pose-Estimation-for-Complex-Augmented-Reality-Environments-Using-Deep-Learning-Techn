#!/bin/bash -l

### User fille in HERE
#
## Required batch arguments
#SBATCH --job-name=WHATEVER
#SBATCH --partition=mp_ib
#SBATCH --ntasks=16
##SBATCH --exclusive
#
## Optional batch arguments (uncomment if used)
##SBATCH --time=2-00:00:00
#
## Suggested batch arguments
##SBATCH --mail-type=ALL
##SBATCH --mail-user=moumed2810@gmail.com
#
## Logging arguments (IMPORTANT)
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err


### Variables Summary
echo ""
echo -e "\033[34m---------------------------------------------------------------------------------------\033[0m"
echo -e "\033[34mVariables Summary: \033[0m"
echo -e "\tWorking Directory = $SLURM_SUBMIT_DIR"
echo -e "\tJob ID = $SLURM_JOB_ID"
echo -e "\tJob Name = $SLURM_JOB_NAME"
echo -e "\tJob Hosts = $SLURM_JOB_NODELIST"
echo -e "\tNumber of Nodes = $SLURM_NNODES"
echo -e "\tNumber of Cores = $SLURM_NTASKS"
echo -e "\tCores per Node = $SLURM_NTASKS_PER_NODE"

### Module Selection
module purge
module load PYTHON/3.6.12


### Send Commands
python fit_3dNet.py

### EOF