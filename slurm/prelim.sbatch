#!/bin/bash
# Preliminary run: first 150 test-split tasks x (3 prelim models x 3 efforts) = 9 array jobs.
# Re-submitting resumes: requests that already succeeded are skipped.
# Submit from the project folder:  sbatch slurm/prelim.sbatch

#SBATCH --account=phd
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0-08:00:00
#SBATCH --array=0-8
#SBATCH --job-name=webshop_prelim
#SBATCH --output=slurm/logs/%x_%A_%a.out

echo "Job $SLURM_JOB_ID array task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
module load python/booth/3.12
source ~/venvs/webshop-vlm/bin/activate

python experiments/run_choice.py \
    --tasks data/tasks/tasks_test.jsonl \
    --group prelim --array-index "$SLURM_ARRAY_TASK_ID" \
    --limit 150 --concurrency 8

echo "Finished at $(date)"
