import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import wraps, partial
from einops import rearrange, repeat
import math
from transformers.modeling_outputs import BaseModelOutput
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sys


def _many(fn):
    @wraps(fn)
    def inner(tensors, pattern, **kwargs):
        return (fn(tensor, pattern, **kwargs) for tensor in tensors)
    return inner

rearrange_many = _many(rearrange)

def exists(val):
    return val is not None

def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, latents):
        """
        einstein notation
        b - batch
        t - time
        n - sequence
        d - dimension
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h = h)

        q = q * self.scale

        # attention

        sim = einsum('... i d, ... j d  -> ... i j', q, k)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h = h)
        return self.to_out(out)

class Compression_Net(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        num_media_embeds = 4,
        ff_mult = 4,
        compress_dim = 64,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.media_pos_emb = nn.Parameter(torch.randn(num_media_embeds, 1, dim))
        self.compress_dim = compress_dim

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = nn.LayerNorm(compress_dim)
        self.fn = nn.Linear(dim, compress_dim)

    def forward(self, x):
        if len(x.shape) == 3:
            x = rearrange(x, 'b n d -> b 1 n d')

        times = x.shape[1]
        x = x + self.media_pos_emb[:times]

        latents = repeat(self.latents, 'n d -> b m n d', b = x.shape[0], m = x.shape[1])

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.fn(latents)
        return self.norm(latents)

# gated cross attention
    
class Reconstruction_Net(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        num_media_embeds = 4,
        ff_mult = 4,
        reconstruct_dim = 1024,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.media_pos_emb = nn.Parameter(torch.randn(num_media_embeds, 1, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = nn.LayerNorm(reconstruct_dim)
        self.fn = nn.Linear(dim, reconstruct_dim)

    def forward(self, x):
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')

        times = x.shape[1]
        x = x + self.media_pos_emb[:times]

        latents = repeat(self.latents, 'n d -> b m n d', b = x.shape[0], m = x.shape[1])

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.fn(latents)
        return self.norm(latents)

# gated cross attention
    
class CombineNet(nn.Module):
    def __init__(self, encoder, compress_net, reconstruct_net, decoder, model, path = None):
        super(CombineNet, self).__init__()

        self.encoder = encoder # molgen encoder
        self.decoder = decoder # molgen decoder
        self.compress_net = compress_net
        self.compress_dim = self.compress_net.compress_dim
        self.reconstruct_net = reconstruct_net
        self.base_model = model

        if path:
            state_dict = torch.load(path)
            self.compress_net.load_state_dict(state_dict['compression_state_dict'], strict=True)
            self.reconstruct_net.load_state_dict(state_dict['reconstruction_state_dict'], strict=True)

    def forward(self, chem_string, attention_mask, decoder_input):
        # Apply ReLU activation after first and second layers
        encoder_output = self.encoder(chem_string, attention_mask)
        compressed_output = self.compress_net(encoder_output['last_hidden_state'])
        reconstructed_output = self.reconstruct_net(compressed_output)
        reconstruct_out = BaseModelOutput(last_hidden_state=reconstructed_output)
        output = self.base_model(decoder_input_ids = decoder_input, encoder_outputs = reconstruct_out).logits.transpose(1,2)
        return output
    
    def compress(self, chem_string, attention_mask, path_of_model = None):
        if path_of_model == None:
            trained_compress = self.compress_net
        else:
            trained_compress = self.compress_net.load_state_dict(torch.load(path_of_model)['compression_state_dict'])
        encoder_output = self.encoder(chem_string, attention_mask)
        return trained_compress(encoder_output['last_hidden_state']) 
    
    def generate_autoregressively(self, encoder_hidden_states, max_length = 1024, **generation_kwargs):
        # Auto-regressive generation method
        # Initialize with start token, e.g., <s>
        # generated = torch.tensor([])        
        generated = torch.tensor([[self.bart.config.decoder_start_token_id]])
        for _ in range(max_length):
            # Forward pass for the current sequence
            outputs = self.output(self.decoder(generated, encoder_hidden_states))
            # Extract logits of the last token
            logits = outputs[:, -1, :]
            # Sample or choose the next token (greedy approach shown here)
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            # Append the next token to the sequence
            generated = torch.cat((generated, next_token), dim=1)
            # print(generated) 
            # Break if EOS token is generated
            if next_token.item() == self.bart.config.eos_token_id:
                break