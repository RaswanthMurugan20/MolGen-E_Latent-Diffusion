"""
Stage 1 entry point — train the molecule compression autoencoder.

Learns the latent space the diffusion model (Stage 2) operates in: a SELFIES string is
encoded by a frozen MolGen (BART) language model, compressed to a small latent by a
Perceiver autoencoder, then decompressed and decoded back to SELFIES. Only the autoencoder
is trained (the language model is frozen); the objective is reconstruction.

The trained checkpoint is written to ``<save_dir>/<dataset>/<timestamp>/`` and is later
passed to ``train_text_diffusion.py`` via ``--latent_model_path``.

See ``scripts/autoencoder/bart_base_roc.sh`` for an example invocation.
"""

import os
import json
import argparse

from latent_models.latent_finetuning import Trainer


def main(args):
    trainer = Trainer(
        args=args,
        dataset_name=args.dataset_name,
        train_batch_size = args.train_batch_size,
        eval_batch_size = args.eval_batch_size,
        train_lr = args.learning_rate,
        train_num_steps = args.num_train_steps,
        lr_schedule = args.lr_schedule,
        num_warmup_steps = args.lr_warmup_steps,
        adam_betas = (args.adam_beta1, args.adam_beta2),
        adam_weight_decay = args.adam_weight_decay,
        eval_every = args.eval_every,
        results_folder = args.output_dir,
        mixed_precision=args.mixed_precision,
    )

    if args.resume_dir:
        trainer.load(args.resume_dir, resume_training=args.resume_training)

    if args.eval:
        trainer.validation(file_path=args.resume_dir)
        return

    trainer.train()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="HUBioDataLab/SELFormer-selfies")
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--enc_dec_model", type=str, default="zjunlp/MolGen-large")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_encoder_latents", type=int, default=32)
    parser.add_argument("--num_decoder_latents", type=int, default=32)
    parser.add_argument("--dim_ae", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--l2_normalize_latents", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="saved_latent_models")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_steps", type=int, default=1)
    parser.add_argument("--lr_schedule", type=str, default="linear")
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--wandb_name", type=str, default=f'bart-roc-l2norm-test-16-8')
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity/team. Defaults to your default wandb account.")
    parser.add_argument(
        "--lm_mode",
        type=str,
        default="freeze",
        choices=["freeze", "ft",],
        help=(
            "How to fine-tune LM."
        ),
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--resume_training", action="store_true", default=False)
    parser.add_argument("--resume_dir", type=str, default=None)

    args = parser.parse_args()

    # When resuming or evaluating, restore the model config from the saved run, but keep the
    # run-specific args (where to log / load from) from the current command line.
    if args.eval or args.resume_training:
        with open(os.path.join(args.resume_dir, 'args.json'), 'rt') as f:
            saved_args = json.load(f)
        args_dict = vars(args)
        heldout_params = {'wandb_name', 'output_dir', 'resume_dir', 'eval'}
        for k, v in saved_args.items():
            if k in heldout_params:
                continue
            args_dict[k] = v

    main(args)