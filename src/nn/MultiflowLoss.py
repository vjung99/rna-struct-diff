import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiflowLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dfm = DiscreteFlowMatching()
        self.cfm_x = ContinuousFlowMatching()
        self.cfm_v = ContinuousFlowMatching()
        self.vectorMap3d = VectorMap3dLoss()

    def forward(self, x_pred, x_1, v_pred, v_1, s_pred, s_1, token_mask, atom_mask, t):
        '''

        '''
        dfm_loss = self.dfm(s_pred, s_1, token_mask)
        strucural_loss = self.vectorMap3d(
            x_pred, v_pred, x_1, v_1, atom_mask, t
        )

        return {
            "loss": dfm_loss + strucural_loss,
            "dfm": dfm_loss,
            "structural_loss": strucural_loss,
        }


class ContinuousFlowMatching(nn.Module):
    """
    We dont use this as it just is MSE between all atom positions.
    VectorMap3dLoss checks the distance between all atoms 1 to 1, Matching the RiboFlow paper and the equivariant nature of our model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred, x_1, x_0, mask):
        return torch.mean(self.mse(pred, x_1 - x_0) * mask)


class VectorMap3dLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, x_pred, v_pred, x_1, v_1, atom_mask, t):
        B, L, _ = x_pred.shape
        x_pred = x_pred.unsqueeze(2).expand(B, L, 24, -1)
        v_pred = v_pred.reshape((B, L, 24, -1))
        x_1 = x_1.unsqueeze(2).expand(B, L, 24, -1)
        v_1 = v_1.reshape((B, L, 24, -1))

        atom_pos_pred = x_pred + v_pred
        atom_pos_true = x_1 + v_1

        d_atom = atom_pos_true - atom_pos_pred

        keep = atom_mask.reshape(B, L, 24, 3).all(-1)
        d_atom = d_atom.reshape(B, -1, 3) * keep.reshape(B, -1, 1)
        N_atoms = keep.reshape(B, -1).sum(-1)

        sq_sum = (d_atom * d_atom).sum(dim=(1, 2))
        sum_d = d_atom.sum(dim=1)

        l_struct = (2 * N_atoms * sq_sum - 2 * (sum_d * sum_d).sum(dim=-1)) / (
            3 * N_atoms**2
        )
        return l_struct.mean()


class DiscreteFlowMatching(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred, s_1, mask):
        per_residue_loss = F.cross_entropy(
            pred, s_1.long() * mask.reshape(-1), reduction="none"
        ) * mask.reshape(-1)
        return torch.mean(per_residue_loss)
