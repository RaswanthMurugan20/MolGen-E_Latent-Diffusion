#!/usr/bin/env bash
# Sample molecules from a trained diffusion model and score them.
# Point --resume_dir at the Stage-2 run you want to evaluate.

python train_text_diffusion.py --eval \
  --resume_dir saved_diff_models/your-stage2-run \
  --sampler ddpm --sampling_schedule cosine \
  --sampling_timesteps 250 --num_samples 1000 --wandb_name eval
