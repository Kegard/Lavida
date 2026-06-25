#!/bin/bash
#SBATCH --job-name=adapt_rvrg
#SBATCH --partition=tamper_resistance
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/data/jindong_gu/LaViDa/logs/adapt_rvrg_%j.log

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lavida
cd /data/jindong_gu/LaViDa

bash AdaptRVRG/run_textvqa_minimal.sh "$@"
