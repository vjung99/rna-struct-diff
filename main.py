import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.datasets import RNACoGenerationDataset
from src.nn.EuclidianNeuralNet import EuclidianNeuralNet
from src.nn.MultiflowLoss import MultiflowLoss
from tqdm import tqdm
import wandb

'''

MAIN training loop. 
TODO: Implement RNAFLOW
  - e3nn architecture -> done
  - data preprocessing -> done
  - Multiflow flow matching with discrete flow matching for sequence generation

'''

DEVICE = 'cpu'
torch.manual_seed(42)

# Written from Multiflow paper
def mask_seq(s: torch.Tensor, x, v, mask_char):
    B, L = s.shape
    s_t = torch.clone(s).to(s.device)
    x_t = torch.clone(x).to(x.device)
    v_t = torch.clone(v).to(v.device)

    # TODO: after Ribogen test try correlated masking (same area of sequence / geometry)
    t = torch.rand(B, device=s.device)  # t sampled from U(0,1)
    # TODO: The noised version of X and V should not be mask tokens but should be Normal distribution for V and maybe exp like (12) in Multiflow for X
    s_t[torch.rand((B, L), device=s.device) < t[:, None]] = (
        mask_char  
    )

    x_0, v_0 = torch.normal(torch.zeros_like(x)), torch.normal(torch.zeros_like(v))

    x_t = x * t[:, None, None] + (1 - t[:, None, None]) * x_0
    v_t = v * t[:, None, None] + (1 - t[:, None, None]) * v_0

    return s_t, x_t, v_t, x_0, v_0, t


def run_epoch(model, optimizer, dataloader, loss_func, wandb_run):
  model.train()
  for s_b, x_b, v_b, token_mask, atom_mask in (pbar := tqdm(dataloader)):
    optimizer.zero_grad()

    s_b = s_b.to(DEVICE)
    x_b = x_b.to(DEVICE)
    v_b = v_b.to(DEVICE)

    s_t, x_t, v_t, x_0, v_0, t = mask_seq(s_b, x_b, v_b, 5) # TODO: Think of how to encode mask token. hardcoded 5 is not ideal obv
    s_pred, x_pred, v_pred = model(s_t, x_t, v_t, t, token_mask)

    loss = loss_func(x_pred, x_b, x_0, v_pred, v_b, v_0, s_pred.reshape((-1, 4)), s_b.reshape(-1), token_mask, atom_mask)

    pbar.set_description(f"Loss: {loss}")
    wandb_run.log({'loss': loss['loss'], 'cfm_x_loss': loss['cfm_x'], 'cfm_v_loss': loss['cfm_v'] , 'dfm': loss['dfm']})


    loss['loss'].backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()

def main(opts):
  df = np.load('./dataset/dataset.npz')
  s = df['S']
  x = df['X']
  v = df['V']


  wandb_run = wandb.init(
      entity="ayynoa",
      project="rna-cogeneration",
      config={
      },
  )

  print(f'Imported dataset {s.shape=} {x.shape=} {v.shape=}')

  model = EuclidianNeuralNet(layers=4)
  loss_func = MultiflowLoss()
  dataset = RNACoGenerationDataset(s, x, v)
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

  for i in range(10):
    run_epoch(model, optimizer, dataloader, loss_func, wandb_run)


if __name__ == "__main__":
  opts = {} # TODO: load cmdline args
  main(opts)
