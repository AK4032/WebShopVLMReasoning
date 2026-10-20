#!/bin/bash
# One-time setup on Mercury. Run from the project folder on a compute node:
#   srun --account=phd --pty bash --login
#   cd ~/webshop_vlm && bash slurm/setup_mercury.sh
set -euo pipefail

module load python/booth/3.12
python3 -m venv ~/venvs/webshop-vlm
source ~/venvs/webshop-vlm/bin/activate
pip install --upgrade pip
pip install "httpx>=0.27" "pyyaml>=6"

mkdir -p slurm/logs results

if [ ! -s ~/.openrouter_key ]; then
    echo "No API key found. Store it (input is hidden) with:"
    echo "  read -s -p 'OpenRouter key: ' k && printf '%s' \"\$k\" > ~/.openrouter_key && chmod 600 ~/.openrouter_key"
fi

python experiments/run_choice.py --tasks data/tasks/tasks_dev.jsonl \
    --model-key gpt-6.1-sol --effort low --limit 1 --dry-run | tail -n 1
echo "Setup complete."
