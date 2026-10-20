# Running the WebShop choice-set experiments on Mercury

Mercury only runs the OpenRouter queries (`experiments/run_choice.py`, Python 3.12).
Building tasks, scoring, and analysis run locally in the Python 3.8 `webshop` env.

## 1. Build and transfer the bundle (local)

```powershell
python experiments/make_bundle.py        # -> dist/mercury_bundle.zip
scp dist/mercury_bundle.zip <BoothID>@mercury.chicagobooth.edu:~/
```

## 2. One-time setup (Mercury)

```bash
mkdir -p ~/webshop_vlm && cd ~/webshop_vlm && unzip -o ~/mercury_bundle.zip
srun --account=phd --pty bash --login      # modules only work on compute nodes
cd ~/webshop_vlm && bash slurm/setup_mercury.sh
read -s -p 'OpenRouter key: ' k && printf '%s' "$k" > ~/.openrouter_key && chmod 600 ~/.openrouter_key
exit                                        # back to the login node
```

Re-transferring a new bundle later only needs the `unzip -o` step; the venv and key persist.

## 3. Submit (Mercury, from `~/webshop_vlm`)

| Script | What it runs | Requests |
|---|---|---|
| `slurm/pilot.sbatch` | 20 robust near-miss dev tasks x 3 models x 3 efforts | 180 |
| `slurm/prelim.sbatch` | 150 robust near-miss test tasks x 3 models x 3 efforts | 1,350 |

```bash
sbatch slurm/pilot.sbatch
squeue --user=$USER                         # monitor
tail -f slurm/logs/webshop_pilot_<jobid>_0.out
```

Array index -> (model, effort) follows `experiments/configs/models.yaml`: index 0-2 is the first
model at low/medium/high, 3-5 the second, 6-8 the third. Re-submitting a script skips requests
that already succeeded, so it is safe after timeouts or failures.

## 4. Bring results back and analyze (local)

```powershell
scp -r <BoothID>@mercury.chicagobooth.edu:~/webshop_vlm/results/raw results/
python experiments/score.py --split dev_robust      # or test_robust
python experiments/analyze.py --split dev_robust
```

Outputs: `results/scored/<split>.csv` (one row per response) and
`results/analysis/<split>/` (summary tables and figures).
