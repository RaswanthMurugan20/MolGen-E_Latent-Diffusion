#!/usr/bin/env bash
# Stage 2 — train the conditional latent diffusion model.
# Point --latent_model_path at the Stage-1 run from scripts/autoencoder/bart_base_roc.sh.
# Each block is one conditioning task; uncomment the one you want to run.

LATENT_MODEL_PATH="saved_latent_models/your-stage1-run"

# --- multi-objective (QED / SA / GSK3B / JNK3), runs on the bundled data ---
python train_text_diffusion.py \
  --task multi-objective --vector_conditional --condition_dim 4 \
  --enc_dec_model zjunlp/MolGen-large \
  --latent_model_path "$LATENT_MODEL_PATH" \
  --tx_dim 512 --tx_depth 12 --num_dense_connections 3 \
  --objective pred_x0 --loss_type l2 --train_schedule cosine \
  --self_condition --scale_shift --train_prob_self_cond 0.5 \
  --sampling_timesteps 80 --learning_rate 2e-4 --train_batch_size 32 \
  --num_train_steps 65000 --save_and_sample_every 750 --num_samples 100 \
  --optimizer adamw --wandb_name multiobj

# --- gene expression (the core MolGene-E setting) ---
# Supply your own gene data via --gene_train_path / --gene_val_path / --gene_test_path,
# and set --condition_dim to your gene-vector size.
# python train_text_diffusion.py \
#   --task phenotype --vector_conditional --condition_dim 2518 \
#   --enc_dec_model zjunlp/MolGen-large \
#   --latent_model_path "$LATENT_MODEL_PATH" \
#   --tx_dim 512 --tx_depth 12 --num_dense_connections 3 \
#   --objective pred_x0 --loss_type l2 --train_schedule cosine \
#   --self_condition --scale_shift --train_prob_self_cond 0.5 \
#   --sampling_timesteps 80 --learning_rate 2e-4 --train_batch_size 32 \
#   --num_train_steps 65000 --save_and_sample_every 1000 --num_samples 300 \
#   --optimizer adamw --wandb_name phenotype

# --- multi-objective DPO fine-tuning (Stage 3) ---
# Pass the trained Stage-2 run via --resume_dir.
# python train_text_diffusion.py \
#   --task dpo_training --vector_conditional --condition_dim 4 \
#   --enc_dec_model zjunlp/MolGen-large \
#   --latent_model_path "$LATENT_MODEL_PATH" \
#   --resume_dir saved_diff_models/your-stage2-run \
#   --tx_dim 512 --tx_depth 12 --num_dense_connections 3 \
#   --objective pred_x0 --loss_type l2 --train_schedule cosine \
#   --self_condition --scale_shift --train_prob_self_cond 0.5 \
#   --sampling_timesteps 80 --learning_rate 2e-4 --train_batch_size 32 \
#   --beta 5000 --num_train_steps 1000 --save_and_sample_every 200 \
#   --optimizer adamw --wandb_name dpo
