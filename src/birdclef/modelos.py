"""Modelos treinados do zero. M1 (CNN) é da Parte 1; a CRNN (M2) e os pré-treinados são da Parte 2.

Todos recebem o log-Mel normalizado com forma (B, n_mels, quadros) e devolvem logits (B, classes).
"""

import torch
from torch import nn


class BlocoConv(nn.Module):
    """Conv 3x3 - BN - ReLU - MaxPool 2x2."""

    def __init__(self, entrada: int, saida: int):
        super().__init__()
        self.camadas = nn.Sequential(
            nn.Conv2d(entrada, saida, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(saida),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.camadas(x)


class CNN(nn.Module):
    """M1: 4 blocos Conv-BN-ReLU-Pool, pooling global e uma camada linear.

    O pooling global faz a média na frequência e combina média e máximo no tempo: o canto pode
    ocupar só um trecho da janela, e o máximo preserva esse trecho.
    """

    def __init__(self, n_classes: int, canais=(32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        blocos, entrada = [], 1
        for c in canais:
            blocos.append(BlocoConv(entrada, c))
            entrada = c
        self.extrator = nn.Sequential(*blocos)
        self.dropout = nn.Dropout(dropout)
        self.saida = nn.Linear(entrada, n_classes)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        h = self.extrator(x.unsqueeze(1))  # (B, C, M', T')
        h = h.mean(dim=2)                  # média na frequência -> (B, C, T')
        return h.mean(dim=2) + h.amax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.saida(self.dropout(self.embedding(x)))


def criar(nome: str, n_classes: int) -> nn.Module:
    if nome == "M1":
        return CNN(n_classes)
    raise ValueError(f"modelo desconhecido: {nome}")


def n_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
