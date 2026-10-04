"""arch/ex_model: EX-KAN (two-anchor excess model), see arch/ex_prereg.md (3938f8f, Amendment 1 fa6ae2a).

delta = log k - anchor_PC = x_co * [ s(eps_co, ln_eta_co, M_co, 1000/T) + (1 - x_co) * g(eps_co, ln_eta_co, M_co, 1000/T, molality, salt) ]
s: kan.KANLayer(4, 3) -> tanh -> kan.KANLayer(3, 1), no embedding; g: kan.KAN(d_num=5, widths=[6, 1]) (7 inputs incl. tanh(salt emb))
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from kan import DELTA_IN, KAN, LR, SMOOTH, WD, KANLayer

S_COLS = [DELTA_IN.index(c) for c in ["eps_co", "lneta_co", "M_co", "invT"]]
G_COLS = [DELTA_IN.index(c) for c in ["eps_co", "lneta_co", "M_co", "invT", "molal"]]


class SNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([KANLayer(4, 3), KANLayer(3, 1)])

    def forward(self, x):
        return self.layers[1](torch.tanh(self.layers[0](x)))[:, 0]

    def smoothness(self):
        return sum(l.smoothness() for l in self.layers)


class EXKAN(nn.Module):
    def __init__(self, n_salt):
        super().__init__()
        self.s = SNet()
        self.g = KAN(len(G_COLS), [6, 1], n_salt)

    def s_of(self, x):
        """s on scaled 8-column final.py inputs."""
        return self.s(x[:, S_COLS])

    def forward(self, b):
        x, xco = b["x"], b["x_co"]
        return xco * (self.s_of(x) + (1 - xco) * self.g(x[:, G_COLS], b["s"])[:, 0])

    def smoothness(self):
        return self.s.smoothness() + self.g.smoothness()


def train_ex(n_salt, b, epochs, seed):
    """AdamW exactly as final.py / kan.train (lr 3e-3, wd 1e-4, full batch, fixed epochs, float32)."""
    torch.manual_seed(seed)
    net = EXKAN(n_salt)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(net(b), b["y"]) + SMOOTH * net.smoothness()
        loss.backward()
        opt.step()
    return net.eval()


def predict_ex(net, b):
    with torch.no_grad():
        return net(b).double().numpy()
