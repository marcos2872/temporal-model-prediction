"""Modelos multivariáveis — cópias fiéis dos notebooks M1/M2/M3.

- DLinearMulti: `multivariavel/notebooks/M1-dlinear-multi.ipynb` §8
- PatchTSTCI:  `multivariavel/notebooks/M2-patchtst-multi-CI.ipynb` §8
- PatchTST_CD: `multivariavel/notebooks/M3-patchtst-multi-CD.ipynb` §8

Forward recebe z-score e devolve z-score (RevIN invertida dentro);
a desnormalização p/ unidade original é feita pelo chamador
com mu/sd de `normalizacao.json` (ver `features.desnorm_saida`).
"""

import torch
import torch.nn as nn

L = 2304
LN, PATCH_P, PATCH_S = 2016, 48, 24  # 83 tokens
D_MODEL, NLAYERS, NHEAD, FF, DROPOUT = 64, 3, 4, 128, 0.1
N_TF = 7  # time-feats por canal no CI: tod_sin/cos + solar + 4 Fourier-origem
POOL_K = 25
DIN = 11  # 4 canais + 3 tempo + 4 fourier-origem


class DLinearMulti(nn.Module):
    def __init__(self, H=288, din=DIN, Lin=L, k=POOL_K):
        super().__init__()
        self.pool = nn.AvgPool1d(k, stride=1, padding=k // 2)
        self.linT_ph = nn.Linear(din * Lin, H)
        self.linS_ph = nn.Linear(din * Lin, H)
        self.linT_od = nn.Linear(din * Lin, H)
        self.linS_od = nn.Linear(din * Lin, H)
        self.gamma = nn.Parameter(torch.ones(4))
        self.beta = nn.Parameter(torch.zeros(4))

    def forward(self, x):
        v = x[:, :4, :]
        mu = v.mean(dim=2, keepdim=True); sg = v.std(dim=2, keepdim=True).clamp_min(1e-3)
        g = self.gamma[None, :, None]; b = self.beta[None, :, None]
        xn = torch.cat([g * (v - mu) / sg + b, x[:, 4:, :]], dim=1)
        t = self.pool(xn); s = xn - t
        tf, sf = t.flatten(1), s.flatten(1)
        yph = self.linT_ph(tf) + self.linS_ph(sf)
        yod = self.linT_od(tf) + self.linS_od(sf)
        yph = (yph - self.beta[1]) / self.gamma[1].clamp_min(1e-3) * sg[:, 1] + mu[:, 1]
        yod = (yod - self.beta[0]) / self.gamma[0].clamp_min(1e-3) * sg[:, 0] + mu[:, 0]
        return torch.stack([yph, yod], dim=2)  # (B, H, 2) [ph, od]


class PatchTSTCI(nn.Module):
    def __init__(self, H, n_tf=N_TF):
        super().__init__()
        self.n_tf = n_tf
        self.N = (LN - PATCH_P) // PATCH_S + 1
        assert self.N == 83, self.N
        self.proj = nn.Linear((1 + n_tf) * PATCH_P, D_MODEL)
        self.pos = nn.Parameter(torch.randn(1, self.N, D_MODEL) * 0.02)
        layer = nn.TransformerEncoderLayer(D_MODEL, NHEAD, FF, DROPOUT, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, NLAYERS)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(self.N * D_MODEL, H)
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, xc):
        v = xc[:, :1, :]
        mu = v.mean(dim=2, keepdim=True)
        sg = v.std(dim=2, keepdim=True).clamp_min(1e-3)
        xn = torch.cat([self.gamma * (v - mu) / sg + self.beta, xc[:, 1:, :]], dim=1)
        w = xn.unfold(2, PATCH_P, PATCH_S).permute(0, 2, 1, 3).reshape(xc.size(0), self.N, -1)
        z = self.proj(w) + self.pos
        z = self.enc(self.drop(z))
        y = self.head(self.drop(z.flatten(1)))
        return (y - self.beta) / self.gamma.clamp_min(1e-3) * sg[:, :, 0] + mu[:, :, 0]


class PatchTST_CD(nn.Module):
    def __init__(self, H=288, din=DIN, Lin=L):
        super().__init__()
        self.N = (Lin - PATCH_P) // PATCH_S + 1
        self.proj = nn.Linear(din * PATCH_P, D_MODEL)
        self.pos = nn.Parameter(torch.randn(1, self.N, D_MODEL) * 0.02)
        layer = nn.TransformerEncoderLayer(D_MODEL, NHEAD, FF, DROPOUT, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, NLAYERS)
        self.drop = nn.Dropout(DROPOUT)
        self.head_ph = nn.Linear(self.N * D_MODEL, H)
        self.head_od = nn.Linear(self.N * D_MODEL, H)
        self.gamma = nn.Parameter(torch.ones(4))
        self.beta = nn.Parameter(torch.zeros(4))

    def forward(self, x):
        v = x[:, :4, :]
        mu = v.mean(dim=2, keepdim=True); sg = v.std(dim=2, keepdim=True).clamp_min(1e-3)
        g = self.gamma[None, :, None]; b = self.beta[None, :, None]
        xn = torch.cat([g * (v - mu) / sg + b, x[:, 4:, :]], dim=1)
        z = self.proj(xn.unfold(2, PATCH_P, PATCH_S).permute(0, 2, 1, 3).flatten(2)) + self.pos
        z = self.enc(self.drop(z))
        f = self.drop(z.flatten(1))
        yph = self.head_ph(f)
        yod = self.head_od(f)
        yph = (yph - self.beta[1]) / self.gamma[1].clamp_min(1e-3) * sg[:, 1] + mu[:, 1]
        yod = (yod - self.beta[0]) / self.gamma[0].clamp_min(1e-3) * sg[:, 0] + mu[:, 0]
        return torch.stack([yph, yod], dim=2)  # (B, H, 2) [ph, od]


FAMILIAS = {"M1": DLinearMulti, "M3": PatchTST_CD}
