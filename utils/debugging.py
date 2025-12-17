import torch
from attn_gym import visualize_attention_scores

def print_mask_debug(mask_mod, qsize, ksize, viz_name="duar"):
  def make_tensor(ctx_size):
      return torch.ones(1, 1, ctx_size, 8, device="cuda")
  query, key = make_tensor(qsize), make_tensor(ksize)
  visualize_attention_scores(
      query, key, mask_mod=mask_mod, device="cuda", name=viz_name
  )