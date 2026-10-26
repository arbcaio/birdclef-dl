"""Figuras dos experimentos (PNG em `resultados/`)."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from birdclef import treino
from birdclef.experimentos import PASTA_RESULTADOS

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
TEXTO, TEXTO_2, GRADE = "#0b0b0b", "#52514e", "#e4e3df"
AZUIS = LinearSegmentedColormap.from_list("azuis", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": TEXTO_2,
    "axes.labelcolor": TEXTO_2, "xtick.color": TEXTO_2, "ytick.color": TEXTO_2,
    "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titlecolor": TEXTO, "figure.dpi": 150,
})

ROTULOS = {"sem": "sem aumento", "specaug": "SpecAugment", "mixup": "mixup", "ruido": "ruído", "todas": "todas"}


def _salvar(fig, caminho: Path):
    fig.tight_layout()
    fig.savefig(caminho, bbox_inches="tight")
    plt.close(fig)


def e2_barras(df: pd.DataFrame, metrica: str = "teste_f1_macro"):
    """F1-macro no teste por configuração: média e desvio das sementes, pontos individuais."""
    ordem = list(ROTULOS)
    g = df.groupby("aumento")[metrica]
    media, desvio = g.mean().reindex(ordem), g.std().reindex(ordem)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(ordem))
    ax.bar(x, media, width=0.6, color=SERIES[0], yerr=desvio, capsize=4, error_kw={"ecolor": TEXTO_2, "lw": 1})
    for i, a in enumerate(ordem):
        v = df.loc[df["aumento"] == a, metrica]
        ax.scatter(np.full(len(v), i), v, s=14, color=TEXTO, zorder=3)
        ax.text(i, media[a] + desvio[a] + 0.01, f"{media[a]:.3f}", ha="center", va="bottom", color=TEXTO, fontsize=9)
    ax.set_xticks(x, [ROTULOS[a] for a in ordem])
    ax.set_ylabel("F1-macro (teste)")
    ax.set_ylim(max(0, media.min() - 0.15), min(1, (media + desvio).max() + 0.06))
    ax.grid(axis="x", visible=False)
    ax.set_title("E2: aumento de dados no M1 (3 sementes)")
    _salvar(fig, PASTA_RESULTADOS / "e2" / "e2_f1_teste.png")


def e2_curvas(semente: int = 42):
    """F1-macro de validação por época, uma linha por configuração."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for cor, a in zip(SERIES, ROTULOS):
        h = pd.read_csv(PASTA_RESULTADOS / "e2" / f"historico_M1_{a}_s{semente}_f0.csv")
        ax.plot(h["epoca"], h["f1_macro"], color=cor, lw=2, label=ROTULOS[a])
    ax.set_xlabel("época")
    ax.set_ylabel("F1-macro (validação)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"E2: curvas de validação (semente {semente})")
    _salvar(fig, PASTA_RESULTADOS / "e2" / "e2_curvas.png")


def e3_linhas(df: pd.DataFrame, metrica: str = "f1_macro"):
    """F1-macro em função da SNR do ruído, com e sem aumento."""
    ordem = ["limpo"] + [f"{s} dB" for s in (20, 10, 5, 0)]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = np.arange(len(ordem))
    for cor, a in zip(SERIES, ["sem", "todas"]):
        g = df[df["aumento"] == a].groupby("teste")[metrica]
        m, s = g.mean().reindex(ordem), g.std().reindex(ordem)
        ax.fill_between(x, m - s, m + s, color=cor, alpha=0.15, lw=0)
        ax.plot(x, m, color=cor, lw=2, marker="o", ms=6, label=ROTULOS[a])
        ax.text(x[-1] + 0.08, m.iloc[-1], f"{ROTULOS[a]}  {m.iloc[-1]:.2f}", va="center", color=TEXTO, fontsize=9)
    ax.set_xticks(x, ordem)
    ax.set_xlim(-0.2, len(ordem) - 0.2 + 0.9)
    ax.set_xlabel("teste (SNR do ruído ESC-50, fold 5)")
    ax.set_ylabel("F1-macro (teste)")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("E3: robustez a ruído de fundo (M1)")
    _salvar(fig, PASTA_RESULTADOS / "e3" / "e3_snr.png")


def matriz(probs: np.ndarray, y: np.ndarray, classes: list[str], nomes: dict, titulo: str, caminho: Path):
    cm = treino.matriz_confusao(y, probs, len(classes))
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap=AZUIS, vmin=0, vmax=1)
    rot = [nomes.get(c, c) for c in classes]
    ax.set_xticks(range(len(classes)), rot, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), rot, fontsize=7)
    ax.grid(False)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if cm_norm[i, j] > 0.5 else TEXTO)
    ax.set_xlabel("predita")
    ax.set_ylabel("verdadeira")
    ax.set_title(titulo)
    fig.colorbar(im, ax=ax, fraction=0.04, label="fração da classe verdadeira")
    _salvar(fig, caminho)


def todas():
    from birdclef import dados
    e2 = pd.read_csv(PASTA_RESULTADOS / "e2" / "e2_execucoes.csv")
    e2_barras(e2)
    e2_curvas()
    pasta_e2 = PASTA_RESULTADOS / "e2"
    classes = json.loads((pasta_e2 / "classes.json").read_text())
    nomes = dados.carregar_metadados().drop_duplicates("primary_label").set_index("primary_label")["common_name"].to_dict()
    melhor = e2.groupby("aumento")["teste_f1_macro"].mean().idxmax()
    y = np.load(pasta_e2 / "y_teste.npy")
    probs = np.load(pasta_e2 / f"probs_teste_M1_{melhor}_s42_f0.npy")
    matriz(probs, y, classes, nomes, f"M1 ({ROTULOS[melhor]}), teste limpo, semente 42", pasta_e2 / "matriz_confusao.png")
    e3_csv = PASTA_RESULTADOS / "e3" / "e3_execucoes.csv"
    if e3_csv.exists():
        e3_linhas(pd.read_csv(e3_csv))


if __name__ == "__main__":
    todas()
