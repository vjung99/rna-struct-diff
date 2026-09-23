import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn.nn.models.gate_points_2101 import Network
from e3nn import o3

import math


class EuclidianNeuralNet(nn.Module):
  def __init__(self, layers) -> None:
    super().__init__()
    self.nucleotide_embedding = nn.Embedding(6, 16)
    # TODO: thouroughly think about and understand values such as max_radius, num_neighbors and num_nodes
    base_kwargs = {
        "irreps_in": o3.Irreps("32x0e + 1x1o + 24x1o"),
        "irreps_hidden": "8x0e + 8x1o + 8x1e + 4x0o", # TODO: This should probably be the same as irreps in and irreps out 
        "irreps_out": "16x0e + 1x1o + 24x1o",
        "irreps_edge_attr": o3.Irreps.spherical_harmonics(3),
        "irreps_node_attr": None,
        "layers": layers,
        "max_radius": 0.8,
        "number_of_basis": 10,
        "radial_layers": 1,
        "radial_neurons": 128,
        "num_neighbors": 8.3,
        "num_nodes": 79.0,
        "reduce_output": False,
    }

    self.base = Network(**base_kwargs)
    self.s_head = o3.Linear("16x0e + 1x1o + 24x1o", "4x0e")
    self.x_head = o3.Linear("16x0e + 1x1o + 24x1o", "1x1o")
    self.v_head = o3.Linear("16x0e + 1x1o + 24x1o", "24x1o")

    for head in (self.s_head, self.x_head, self.v_head):
      nn.init.zeros_(head.weight) 

  def _embed_t(self, timesteps, embedding_dim, max_positions=2000):
    # From https://github.com/jasonkyuyim/multiflow/blob/main/multiflow/models/utils.py#L49
    # They actually got it from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

  def forward(self, s, x, v, t, token_mask):
    B, L = s.shape
    keep = token_mask.bool()

    t_features = self._embed_t(t, 16, max_positions=2056).unsqueeze(1).expand(-1, L, -1).to(s.device)   # (B, L, 16) 

    s_embed = self.nucleotide_embedding(s.long())
    features = torch.concat((s_embed, t_features, x, v), dim=-1)

    s_embed = s_embed[keep]
    features = features[keep]
    batch = torch.arange(B, device = s.device).repeat_interleave(L)[keep.flatten()]

    data = {"pos": x[keep], "x": features, "batch": batch}

    res = self.base(data)
    
    s_res = self.s_head(res)   # [<B, 4] 
    x_res = self.x_head(res)   # [<B, 3] 
    v_res = self.v_head(res)   # [<B, 72] 

    res = torch.concat((s_res, x_res, v_res), dim=-1)

    out = torch.zeros(B, L, res.shape[-1], device=s.device, dtype=res.dtype)
    out[keep] = res
    out = out.reshape(B, L, -1)

    s_pred = out[:,:,:4]
    x_pred = out[:,:,4:7]
    v_pred = out[:,:,7:]

    return s_pred, x_pred, v_pred
