#!/bin/bash
# Smoke test: 3 dev tasks x every (model, effort) pair in the full study = 29 array jobs, 87 requests.
# Checks that each model accepts the request format before the full run.
# Submit from the project folder:  sbatch slurm/smoke.sbatch
# Array index -> model/effort:  python experiments/run_choice.py --group full --list-combos

#SBATCH --account=phd
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0-01:00:00
#SBATCH --array=0-28
#SBATCH --job-name=webshop_smoke
#SBATCH --output=slurm/logs/%x_%A_%a.out

echo "Job $SLURM_JOB_ID array task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
module load python/booth/3.12
source ~/venvs/webshop-vlm/bin/activate

python experiments/run_choice.py \
    --tasks data/tasks/tasks_dev_final.jsonl \
    --group full --array-index "$SLURM_ARRAY_TASK_ID" \
    --limit 3 --concurrency 3

echo "Finished at $(date)"
