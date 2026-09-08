import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiflowLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dfm = DiscreteFlowMatching()
        self.cfm_x = ContinuousFlowMatching()
        self.cfm_v = ContinuousFlowMatching()

    def forward(self, x_pred, x_1, x_0, v_pred, v_1, v_0, s_pred, s_1, token_mask, atom_mask):
        dfm_loss = self.dfm(s_pred, s_1, token_mask)
        cfm_x_loss = self.cfm_x(x_pred, x_1, x_0, token_mask.unsqueeze(-1))
        cfm_v_loss = self.cfm_v(v_pred, v_1, v_0, atom_mask)
        return {"loss": dfm_loss + cfm_x_loss + cfm_v_loss, "dfm": dfm_loss, "cfm_x": cfm_x_loss, 'cfm_v': cfm_v_loss}

class ContinuousFlowMatching(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.mse = nn.MSELoss(reduction='none')

  def forward(self, pred, x_1, x_0, mask):
    return torch.mean(self.mse(pred, x_1 - x_0) * mask)

class DiscreteFlowMatching(nn.Module):
  def __init__(self) -> None:
    super().__init__()

  def forward(self, pred, s_1, mask):
    per_residue_loss = F.cross_entropy(pred, s_1.long() * mask.reshape(-1), reduction='none') * mask.reshape(-1)
    return torch.mean(per_residue_loss)
    
