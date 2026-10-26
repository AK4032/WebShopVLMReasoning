#!/bin/bash
# Full study: all 496 test tasks x every (model, effort) pair = 29 array jobs (~14,400 requests).
# Re-submitting resumes: requests that already succeeded are skipped.
# Submit from the project folder:  sbatch slurm/full.sbatch
# Array index -> model/effort:  python experiments/run_choice.py --group full --list-combos

#SBATCH --account=phd
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=1-12:00:00
#SBATCH --array=0-28
#SBATCH --job-name=webshop_full
#SBATCH --output=slurm/logs/%x_%A_%a.out

echo "Job $SLURM_JOB_ID array task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
module load python/booth/3.12
source ~/venvs/webshop-vlm/bin/activate

python experiments/run_choice.py \
    --tasks data/tasks/tasks_test_final.jsonl \
    --group full --array-index "$SLURM_ARRAY_TASK_ID" \
    --concurrency 6

echo "Finished at $(date)"
