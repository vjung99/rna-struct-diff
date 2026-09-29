"""
MAIN training loop. 
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb

from src.constants import MASK_IDX, N_ATOMS, NUM_BASES, PAD_IDX
from pdb_utils import save_pdb
from src.data.datasets import BatchSampler, RNACoGenerationDataset
from src.data.sec_utils import Sequence
from src.nn.EuclidianNeuralNet import EuclidianNeuralNet
from src.nn.MultiflowLoss import MultiflowLoss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)


def centered_noise(shape):
    noise = torch.normal(torch.zeros(shape, device=DEVICE))
    return noise - noise.mean(dim=1, keepdim=True)


@torch.no_grad()
def sample_from_model(s_0, x_0, v_0, model, steps=200, epsilon=1e-2):
    def unmask_seq_random(s_0, s_logits, dt, t):
        reveal = (dt / (1 - t)).clamp(max=1.0)
        draw = torch.rand(s_0.shape, device=s_0.device) < reveal
        tokens = torch.multinomial(
            F.softmax(s_logits, dim=-1).flatten(0, 1), 1
        ).reshape(s_0.shape)
        return torch.where(draw & (s_0 == MASK_IDX), tokens, s_0)

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

    # Force any remaining position to unmask
    s_0 = torch.where(s_0 >= NUM_BASES, s_logits.argmax(dim=-1), s_0)
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
    s_t[torch.rand((B, L), device=s.device) < t[:, None]] = mask_char

    v_0 = torch.normal(torch.zeros_like(v)).to(s.device)
    x_0 = centered_noise(x.shape).to(s.device)

    x_t = x * t[:, None, None] + (1 - t[:, None, None]) * x_0
    v_t = v * t[:, None, None] + (1 - t[:, None, None]) * v_0

    return s_t, x_t, v_t, x_0, v_0, t


def run_epoch(model, optimizer, dataloader, loss_func, wandb_run):
    model.train()
    for s_b, x_b, v_b, atom_mask in (pbar := tqdm(dataloader)):
        optimizer.zero_grad()

        s_b = s_b.to(DEVICE)
        x_b = x_b.to(DEVICE)
        v_b = v_b.to(DEVICE)
        atom_mask = atom_mask.to(DEVICE)

        token_mask = s_b != PAD_IDX

        s_t, x_t, v_t, x_0, v_0, t = mask_seq(
            s_b, x_b, v_b, MASK_IDX
        )
        s_pred, x_pred, v_pred = model(s_t, x_t, v_t, t, token_mask)

        loss = loss_func(
            x_pred,
            x_b,
            v_pred,
            v_b,
            s_pred.reshape((-1, NUM_BASES)),
            s_b.reshape(-1),
            token_mask,
            atom_mask,
            t,
        )

        losses = {k:v.item() for (k, v) in loss.items()}
        pbar.set_description(f"Loss: {losses}")

        loss["loss"].backward()
        gn = nn.utils.get_total_norm(model.parameters())
        wandb_run.log({**loss, "lr": optimizer.param_groups[0]["lr"], "grad_norm": gn})
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()


def split_stats(name, sequences, crop_len=256):
    """One line per split: unique sequences, structures, cropped nodes, length range."""
    lengths = np.array([seq.length for seq in sequences])
    n_struct = sum(len(seq) for seq in sequences)
    nodes = int(np.minimum(lengths, crop_len).sum())
    print(
        f"{name}: {len(sequences)} unique sequences, {n_struct} structures, {nodes} nodes, "
        f"residues {lengths.min()}-{lengths.max()} (median {int(np.median(lengths))})"
    )


def train(model, dataloader, args):
    wandb_run = wandb.init(
        entity="ayynoa",
        project="rna-cogeneration",
        config={
            "lr": 1e-3,
            "layers": args.layers,
            "sampler": "batch",
            "max_nodes_batch": 3000,
            "crop_len": 256,
            "epochs": args.epochs,
        },
    )

    loss_func = MultiflowLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    for i in range(args.epochs):
        if hasattr(dataloader.batch_sampler, "set_epoch"):
            dataloader.batch_sampler.set_epoch(i)
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
    t.add_argument("-d", "--dataset", default="./data/processed.pt")
    t.add_argument("--split", default="./data/das_split.pt")

    # Inference
    i = sub.add_parser("inference")
    i.add_argument("-c", "--checkpoint", default="./e3nn_checkpoint.pt")
    i.add_argument("-l", "--layers", type=int, default=4)
    i.add_argument("-s", "--steps", type=int, default=200)
    i.add_argument("--length", type=int, default=70)
    i.add_argument("--samples", type=int, default=10)
    i.add_argument('-o', '--out', default='./samples')

    return parser.parse_args()


if __name__ == "__main__":
    args = getargs()

    model = EuclidianNeuralNet(layers=args.layers).to(DEVICE)

    match args.cmd:
        case "train":
            sequences = Sequence.load_sequences_from_file(args.dataset)
            train_idx, val_idx, test_idx = torch.load(args.split, weights_only=False)
            for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
                split_stats(name, [sequences[i] for i in idx])
            dataset = RNACoGenerationDataset([sequences[i] for i in train_idx])

            batch_sampler = BatchSampler(dataset.node_counts)
            dataloader = DataLoader(
                dataset, num_workers=0, batch_sampler=batch_sampler
            )
            train(model, dataloader, args)
        case "inference":
            model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
            s_0 = torch.full((args.samples, args.length), MASK_IDX, dtype=torch.long, device=DEVICE)
            x_0 = centered_noise((args.samples, args.length, 3)).to(DEVICE)
            v_0 = torch.randn(args.samples, args.length, N_ATOMS * 3, device=DEVICE)
            s, x, v = sample_from_model(s_0, x_0, v_0, model, steps=args.steps)
            s_np = s.cpu().numpy()
            x_A = x.cpu().numpy()
            v_A = v.cpu().numpy()
            padded = s_np == PAD_IDX
            x_A[padded] = np.nan
            v_A[padded] = np.nan
            s_np[padded] = 0
            np.savez(os.path.join(args.out, "samples.npz"), S=s_np, X=x_A, V=v_A)
            print("masks left:", int((s == MASK_IDX).sum()))
            for k in range(args.samples):
                # token 4 ('N') is not writable as a residue name
                writable = s_np[k] < NUM_BASES - 1
                print(f"sample {k}: {int((~writable).sum())} unknown residues dropped")
                save_pdb(
                    f"{args.out}/sample_{k}.pdb",
                    s_np[k][writable],
                    x_A[k][writable],
                    v_A[k][writable],
                )
