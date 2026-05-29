#!/usr/bin/env bash
# Stage 1 — train the SELFIES compression autoencoder.
# Single-GPU; prepend `accelerate launch` instead of `python` for multi-GPU.

python train_latent_model.py \
  --dataset_name HUBioDataLab/SELFormer-selfies \
  --enc_dec_model zjunlp/MolGen-large \
  --num_encoder_latents 16 --num_decoder_latents 16 --dim_ae 8 --num_layers 3 \
  --l2_normalize_latents \
  --learning_rate 1e-4 --lr_warmup_steps 1000 --train_batch_size 1 \
  --eval_every 1000 --wandb_name molgen-ae
