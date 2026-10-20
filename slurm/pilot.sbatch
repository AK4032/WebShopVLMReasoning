#!/bin/bash
# Pilot: 20 robust near-miss dev tasks x (3 prelim models x 3 efforts) = 9 array jobs, 180 requests.
# Submit from the project folder:  sbatch slurm/pilot.sbatch

#SBATCH --account=phd
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0-02:00:00
#SBATCH --array=0-8
#SBATCH --job-name=webshop_pilot
#SBATCH --output=slurm/logs/%x_%A_%a.out

echo "Job $SLURM_JOB_ID array task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
module load python/booth/3.12
source ~/venvs/webshop-vlm/bin/activate

python experiments/run_choice.py \
    --tasks data/tasks/tasks_dev_robust.jsonl \
    --group prelim --array-index "$SLURM_ARRAY_TASK_ID" \
    --limit 20 --concurrency 8

echo "Finished at $(date)"
