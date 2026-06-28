#!/bin/bash
set -euo pipefail

cd /data/jindong_gu/LaViDa
sleep 2h
sbatch M3CoT/PostVRG/submit_m3cot_highvisual_refill_vrg_logits.sh
