from torch import nn
import torch.nn.functional as F

from dataset import N_ATOMS


class MultiflowLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dfm = DiscreteFlowMatching()
        self.vectorMap3d = VectorMap3dLoss()

    def forward(self, x_pred, x_1, v_pred, v_1, s_pred, s_1, token_mask, atom_mask, t):
        dfm_loss = self.dfm(s_pred, s_1, token_mask)
        strucural_loss = self.vectorMap3d(
            x_pred, v_pred, x_1, v_1, atom_mask, t
        )

        return {
            "loss": dfm_loss + strucural_loss,
            "dfm": dfm_loss,
            "structural_loss": strucural_loss,
        }


class VectorMap3dLoss(nn.Module):
    """
    MSE between all atom positions, 1-to-1, matching RiboGen eq. 7.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, x_pred, v_pred, x_1, v_1, atom_mask, t):
        B, L, _ = x_pred.shape
        x_pred = x_pred.unsqueeze(2).expand(B, L, N_ATOMS, -1)
        v_pred = v_pred.reshape((B, L, N_ATOMS, -1))
        x_1 = x_1.unsqueeze(2).expand(B, L, N_ATOMS, -1)
        v_1 = v_1.reshape((B, L, N_ATOMS, -1))

        atom_pos_pred = x_pred + v_pred
        atom_pos_true = x_1 + v_1

        d_atom = atom_pos_true - atom_pos_pred

        keep = atom_mask.reshape(B, L, N_ATOMS, 3).all(-1)
        d_atom = d_atom.reshape(B, -1, 3) * keep.reshape(B, -1, 1)
        N_atoms = keep.reshape(B, -1).sum(-1)

        sq_sum = (d_atom * d_atom).sum(dim=(1, 2))
        sum_d = d_atom.sum(dim=1)

        l_struct = (2 * N_atoms * sq_sum - 2 * (sum_d * sum_d).sum(dim=-1)) / (
            3 * N_atoms.clamp(min=1) ** 2
        )
        w = N_atoms.float()
        valid = N_atoms > 0

        return (l_struct[valid] * w[valid]).sum() / w[valid].sum()


class DiscreteFlowMatching(nn.Module):
    """
    Masked cross entropy over residue tokens.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred, s_1, mask):
        m = mask.reshape(-1).bool()

        return F.cross_entropy(pred[m], s_1.long().reshape(-1)[m], reduction="mean")
