import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb

from pdb_utils import save_pdb
from src.data.datasets import RNACoGenerationDataset
from src.nn.EuclidianNeuralNet import EuclidianNeuralNet
from src.nn.MultiflowLoss import MultiflowLoss

"""

MAIN training loop. 
TODO: Implement RNAFLOW
  - e3nn architecture -> done
  - data preprocessing -> done
  - Multiflow flow matching with discrete flow matching for sequence generation

"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MASK_TOKEN = 5
torch.manual_seed(42)


def centered_noise(shape):
    noise = torch.normal(torch.zeros(shape))
    return noise - noise.mean(dim=1, keepdim=True)


@torch.no_grad()
def sample_from_model(s_0, x_0, v_0, model, steps=200, epsilon=1e-2):
    def unmask_seq_random(s_0, s_logits, dt, t):
        reveal = (dt / (1 - t)).clamp(max=1.0)
        draw = torch.rand(s_0.shape, device=s_0.device) < reveal
        tokens = torch.multinomial(
            F.softmax(s_logits, dim=-1).flatten(0, 1), 1
        ).reshape(s_0.shape)
        return torch.where(draw & (s_0 == MASK_TOKEN), tokens, s_0)

    B, L = s_0.shape

    prev_t = 0
    s_logits = None
    token_mask = torch.ones(B, L, dtype=torch.bool, device=s_0.device)
    for t in torch.linspace(epsilon, 1 - epsilon, steps, device=s_0.device):
        dt = t - prev_t
        t_b = t.expand(B)
        s_logits, x_hat, v_hat = model(s_0, x_0, v_0, t_b, token_mask)
        x_0 += dt * (x_hat - x_0) / (1 - t)
        v_0 += dt * (v_hat - v_0) / (1 - t)
        s_0 = unmask_seq_random(s_0, s_logits, dt, t)
        prev_t = t

    s_0 = torch.where(s_0 == MASK_TOKEN, s_logits.argmax(dim=-1), s_0)
    return s_0, x_0, v_0


# Written from Multiflow paper
def mask_seq(s: torch.Tensor, x, v, mask_char, epsilon=1e-2):
    B, L = s.shape
    s_t = torch.clone(s).to(s.device)
    x_t = torch.clone(x).to(x.device)
    v_t = torch.clone(v).to(v.device)

    # TODO: after Ribogen test try correlated masking (same area of sequence / geometry)
    t = torch.rand(B, device=s.device)  # t sampled from U(0,1)
    t = t * (1 - 2 * epsilon) + epsilon
    # TODO: The noised version of X and V should not be mask tokens but should be Normal distribution for V and maybe exp like (12) in Multiflow for X
    s_t[torch.rand((B, L), device=s.device) < t[:, None]] = mask_char

    v_0 = torch.normal(torch.zeros_like(v)).to(s.device)
    x_0 = centered_noise(x.shape).to(s.device)

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
        token_mask = token_mask.to(DEVICE)
        atom_mask = atom_mask.to(DEVICE)

        s_t, x_t, v_t, x_0, v_0, t = mask_seq(
            s_b, x_b, v_b, MASK_TOKEN
        )
        s_pred, x_pred, v_pred = model(s_t, x_t, v_t, t, token_mask)

        loss = loss_func(
            x_pred,
            x_b,
            v_pred,
            v_b,
            s_pred.reshape((-1, 4)),
            s_b.reshape(-1),
            token_mask,
            atom_mask,
            t,
        )

        losses = {k:v.item() for (k, v) in loss.items()}
        pbar.set_description(f"Loss: {losses}")
        wandb_run.log(loss)

        loss["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()


def train(model, dataloader, args):
    wandb_run = wandb.init(
        entity="ayynoa",
        project="rna-cogeneration",
        config={},
    )

    print(f"Imported dataset {s.shape=} {x.shape=} {v.shape=}")

    loss_func = MultiflowLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    for i in range(args.epochs):
        run_epoch(model, optimizer, dataloader, loss_func, wandb_run)
        scheduler.step()
        torch.save(model.state_dict(), args.checkpoint)


def getargs():
    parser = argparse.ArgumentParser("RNA Co-Generation")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Training
    t = sub.add_parser("train")
    t.add_argument("-c", "--checkpoint", default="./e3nn_checkpoint.pt")
    t.add_argument("-e", "--epochs", type=int, default=10)
    t.add_argument("-b", "--batch_size", type=int, default=32)
    t.add_argument("-l", "--layers", type=int, default=4)
    t.add_argument("-d", "--dataset", default="./dataset/dataset.npz")

    # Inference
    i = sub.add_parser("inference")
    i.add_argument("-c", "--checkpoint", default="./e3nn_checkpoint.pt")
    i.add_argument("-l", "--layers", type=int, default=4)
    i.add_argument("-s", "--steps", type=int, default=200)
    i.add_argument("--length", type=int, default=70)
    i.add_argument("--samples", type=int, default=10)
    i.add_argument('-o', '--out', default='./samples')
    i.add_argument("-d", "--dataset", default="./dataset/dataset.npz")

    return parser.parse_args()


if __name__ == "__main__":
    args = getargs()

    model = EuclidianNeuralNet(layers=args.layers).to(DEVICE)

    match args.cmd:
        case "train":
            df = np.load(args.dataset)
            s = df["S"]
            x = df["X"]
            v = df["V"]
            dataset = RNACoGenerationDataset(s, x, v)
            dataloader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
            )
            train(model, dataloader, args)
        case "inference":
            model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
            s_0 = torch.full((args.samples, args.length), MASK_TOKEN, dtype=torch.long, device=DEVICE)
            x_0 = centered_noise((args.samples, args.length, 3)).to(DEVICE)
            v_0 = torch.randn(args.samples, args.length, 72, device=DEVICE)
            s, x, v = sample_from_model(s_0, x_0, v_0, model, steps=args.steps)
            df = np.load(args.dataset)
            x_A = x.cpu().numpy() * float(df["x_scale"])
            v_A = v.cpu().numpy() * float(df["v_scale"])
            np.savez(
                args.out,
                S=s.cpu().numpy(),
                X=x_A,
                V=v_A,
            )
            print("masks left:", int((s == MASK_TOKEN).sum()))
            for k in range(args.samples):
                save_pdb(f"{args.out}/sample_{k}.pdb", s[k], x_A[k], v_A[k])
