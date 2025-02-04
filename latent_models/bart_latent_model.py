import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

from transformers.models.bart.modeling_bart import (
    BartForConditionalGeneration,
)
from latent_models.perceiver_ae import PerceiverAutoEncoder
from latent_models.my_latent_model import Compression_Net, Reconstruction_Net
from einops import rearrange
import sys

class BARTForConditionalGenerationLatent(BartForConditionalGeneration):
    def __init__(self, config, num_encoder_latents, num_decoder_latents, dim_ae, num_layers=2, l2_normalize_latents=False):
        super().__init__(config)
        self.num_encoder_latents = num_encoder_latents
        self.dim_ae = dim_ae
        self.l2_normalize_latents = l2_normalize_latents
        # config.d_model
        self.perceiver_ae = PerceiverAutoEncoder(dim_lm=config.d_model, num_encoder_latents=num_encoder_latents, num_decoder_latents=num_decoder_latents, dim_ae=dim_ae, depth=num_layers, transformer_decoder=True, l2_normalize_latents=l2_normalize_latents)
        # self.cnet = Compression_Net(dim = config.d_model, depth = 3,dim_head = 64,heads = 8, num_latents = num_encoder_latents,num_media_embeds = 4,ff_mult = 4,compress_dim = 8)
        # self.rnet = Reconstruction_Net(dim = 64,depth = 3,dim_head = 64,heads = 8,num_latents = num_decoder_latents, num_media_embeds = 4,ff_mult = 4,reconstruct_dim = config.d_model)

    def get_diffusion_latent(self, encoder_outputs, attention_mask, path=None):
        hidden_state = encoder_outputs[0]
        latent = self.perceiver_ae.encode(hidden_state, attention_mask.bool())
        return latent
        
    def get_decoder_input(self, diffusion_latent):
        return self.perceiver_ae.decode(diffusion_latent)
    
    # Map encoder outputs to decoder inputs
    def encoder_output_to_decoder_input(self, encoder_outputs, attention_mask):
        diffusion_latent = self.get_diffusion_latent(encoder_outputs, attention_mask)
        encoder_outputs['last_hidden_state'] = self.get_decoder_input(diffusion_latent)
        return encoder_outputs
    
