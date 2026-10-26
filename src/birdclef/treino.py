"""Treino e avaliação, comuns a todos os modelos.

- Amostragem: cada época sorteia janelas com peso 1/(janelas da gravação), para que toda gravação
  pese igual, por mais longa que seja.
- Perda: entropia cruzada com rótulos suaves (necessária para o mixup); AdamW; cosseno na taxa de
  aprendizado; precisão mista na GPU.
- Parada antecipada pelo F1-macro de validação, medido por gravação.
- Predição de uma gravação: média das probabilidades das suas janelas.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score

from birdclef import aumento, modelos
from birdclef.aumento import ConfigAumento


@dataclass
class ConfigTreino:
    modelo: str = "M1"
    epocas: int = 30
    paciencia: int = 6
    lote: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    semente: int = 42


@dataclass
class Resultado:
    modelo: torch.nn.Module
    classes: list[str]
    norm: tuple[float, float]
    historico: pd.DataFrame
    metricas_val: dict = field(default_factory=dict)
    segundos: float = 0.0


def dispositivo() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fixar_semente(semente: int) -> None:
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)


# ----------------------------------------------------------------------------- métricas

def metricas(y: np.ndarray, probs: np.ndarray) -> dict:
    """Métricas por gravação. `y`: índices das classes; `probs`: (n, classes)."""
    pred = probs.argmax(1)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    onehot = np.eye(probs.shape[1])[y]
    return {
        "f1_macro": f1_score(y, pred, average="macro"),
        "acc_top1": float((pred == y).mean()),
        "acc_top3": float((top3 == y[:, None]).any(1).mean()),
        "map": average_precision_score(onehot, probs, average="macro"),
    }


def matriz_confusao(y: np.ndarray, probs: np.ndarray, n_classes: int) -> np.ndarray:
    return confusion_matrix(y, probs.argmax(1), labels=range(n_classes))


# ----------------------------------------------------------------------------- predição

@torch.no_grad()
def prever_janelas(modelo, X, idx: np.ndarray, norm, lote: int = 256) -> np.ndarray:
    modelo.eval()
    disp = next(modelo.parameters()).device
    media, desvio = norm
    saidas = []
    for i in range(0, len(idx), lote):
        x = torch.from_numpy(np.asarray(X[idx[i:i + lote]], dtype=np.float32)).to(disp)
        with torch.autocast(disp.type, enabled=disp.type == "cuda"):
            saidas.append(torch.softmax(modelo((x - media) / desvio).float(), 1).cpu().numpy())
    return np.concatenate(saidas) if saidas else np.empty((0, 0))


def prever_gravacoes(modelo, X, janelas: pd.DataFrame, idx: np.ndarray, norm, classes) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Probabilidades por gravação (média das janelas), rótulos e nomes dos arquivos."""
    probs = prever_janelas(modelo, X, idx, norm)
    df = pd.DataFrame(probs, index=janelas["filename"].to_numpy()[idx])
    por_grav = df.groupby(level=0, sort=False).mean()
    rotulo = janelas.iloc[idx].groupby("filename", sort=False)["primary_label"].first().loc[por_grav.index]
    y = rotulo.map({c: i for i, c in enumerate(classes)}).to_numpy()
    return por_grav.to_numpy(), y, por_grav.index


# ----------------------------------------------------------------------------- treino

def indices(janelas: pd.DataFrame, fold_val: int) -> tuple[np.ndarray, np.ndarray]:
    treino = (janelas["particao"] == "treino") & (janelas["fold"] != fold_val)
    val = (janelas["particao"] == "treino") & (janelas["fold"] == fold_val)
    return np.flatnonzero(treino.to_numpy()), np.flatnonzero(val.to_numpy())


def estatisticas(X, idx: np.ndarray, n: int = 3000, semente: int = 0) -> tuple[float, float]:
    """Média e desvio globais (em dB) de uma amostra das janelas de treino."""
    amostra = np.random.default_rng(semente).choice(idx, size=min(n, len(idx)), replace=False)
    x = np.asarray(X[np.sort(amostra)], dtype=np.float32)
    return float(x.mean()), float(x.std())


def treinar(
    X: np.ndarray,
    janelas: pd.DataFrame,
    fold_val: int,
    cfg_aum: ConfigAumento = aumento.SEM,
    cfg: ConfigTreino = ConfigTreino(),
    R: np.ndarray | None = None,
    verbose: bool = True,
) -> Resultado:
    """Treina um modelo com o fold `fold_val` como validação e os demais folds de treino.

    `X` deve estar em memória (np.ndarray) para o sorteio de lotes ser rápido. `R` são os log-Mel
    dos ruídos de treino, obrigatórios se `cfg_aum.ruido`.
    """
    fixar_semente(cfg.semente)
    disp = dispositivo()
    classes = sorted(janelas["primary_label"].unique())
    alvo = janelas["primary_label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    idx_tr, idx_val = indices(janelas, fold_val)
    norm = estatisticas(X, idx_tr)
    media, desvio = norm

    janelas_por_grav = janelas.iloc[idx_tr].groupby("filename")["filename"].transform("size").to_numpy()
    pesos = torch.tensor(1.0 / janelas_por_grav, dtype=torch.double)
    R_t = torch.from_numpy(np.asarray(R, dtype=np.float32)).to(disp) if cfg_aum.ruido else None

    modelo = modelos.criar(cfg.modelo, len(classes)).to(disp)
    otim = torch.optim.AdamW(modelo.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    agenda = torch.optim.lr_scheduler.CosineAnnealingLR(otim, T_max=cfg.epocas)
    escala = torch.amp.GradScaler(enabled=disp.type == "cuda")

    melhor, melhor_estado, sem_melhora, hist = -1.0, None, 0, []
    inicio = time.time()
    for epoca in range(1, cfg.epocas + 1):
        modelo.train()
        sorteio = idx_tr[torch.multinomial(pesos, len(idx_tr), replacement=True).numpy()]
        perda_total = 0.0
        for i in range(0, len(sorteio), cfg.lote):
            lote = sorteio[i:i + cfg.lote]
            x = torch.from_numpy(X[lote].astype(np.float32)).to(disp, non_blocking=True)
            y = F.one_hot(torch.from_numpy(alvo[lote]), len(classes)).float().to(disp)

            if cfg_aum.ruido:
                com_ruido = torch.rand(len(lote), device=disp) < cfg_aum.prob_ruido
                if com_ruido.any():
                    r = R_t[torch.randint(len(R_t), (int(com_ruido.sum()),), device=disp)]
                    snr = torch.empty(len(r), device=disp).uniform_(cfg_aum.snr_min, cfg_aum.snr_max)
                    x[com_ruido] = aumento.misturar_ruido_mel(x[com_ruido], r, snr)
            x = (x - media) / desvio
            if cfg_aum.specaugment:
                x = aumento.spec_augment(x, cfg_aum)
            if cfg_aum.mixup:
                x, y = aumento.mixup(x, y, cfg_aum.alfa_mixup)

            with torch.autocast(disp.type, enabled=disp.type == "cuda"):
                logits = modelo(x)
            perda = -(y * F.log_softmax(logits.float(), 1)).sum(1).mean()
            otim.zero_grad(set_to_none=True)
            escala.scale(perda).backward()
            escala.step(otim)
            escala.update()
            perda_total += perda.item() * len(lote)
        agenda.step()

        probs, y_val, _ = prever_gravacoes(modelo, X, janelas, idx_val, norm, classes)
        m = metricas(y_val, probs)
        hist.append({"epoca": epoca, "perda": perda_total / len(sorteio), **m})
        if verbose:
            print(f"época {epoca:2d}  perda {hist[-1]['perda']:.3f}  F1 val {m['f1_macro']:.3f}  top1 {m['acc_top1']:.3f}")
        if m["f1_macro"] > melhor:
            melhor, sem_melhora = m["f1_macro"], 0
            melhor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            sem_melhora += 1
            if sem_melhora >= cfg.paciencia:
                break

    modelo.load_state_dict(melhor_estado)
    historico = pd.DataFrame(hist)
    metricas_val = historico.loc[historico["f1_macro"].idxmax()].to_dict()
    return Resultado(modelo, classes, norm, historico, metricas_val, time.time() - inicio)


def salvar(res: Resultado, caminho: Path, **extra) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"estado": res.modelo.state_dict(), "classes": res.classes, "norm": res.norm, **extra}, caminho)


def avaliar_teste(res: Resultado, X_teste, janelas_teste: pd.DataFrame) -> tuple[dict, np.ndarray, np.ndarray]:
    """Métricas no teste. `X_teste` e `janelas_teste` contêm só as janelas de teste."""
    idx = np.arange(len(janelas_teste))
    probs, y, _ = prever_gravacoes(res.modelo, X_teste, janelas_teste, idx, res.norm, res.classes)
    return metricas(y, probs), y, probs
