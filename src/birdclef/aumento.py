"""Aumento de dados e ruído de fundo.

Três técnicas, aplicadas só no treino e sobre os log-Mel já pré-computados:

- SpecAugment: máscaras de frequência e de tempo;
- mixup: combinação convexa de pares de exemplos e de seus rótulos;
- ruído de fundo: soma de um ruído da ESC-50 com SNR sorteada.

O ruído do treino é somado no domínio de potência dos log-Mel (sinal e ruído descorrelacionados
somam potência), o que evita refazer o espectrograma a cada época. A SNR é medida sobre a
potência Mel da janela. Para o teste de robustez (E3) o ruído é somado na forma de onda, que é o
caso real, em `preparar_teste_ruidoso`.

A ESC-50 tem 5 folds: os folds 1 a 4 servem ao treino e o fold 5 só ao teste, para que o modelo
nunca veja no treino o ruído com que é testado.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from birdclef import dados, preproc
from birdclef.preproc import ConfigMel, MEL128_32K

FOLD_RUIDO_TESTE = 5
SNRS_TESTE = (20, 10, 5, 0)


@dataclass(frozen=True)
class ConfigAumento:
    specaugment: bool = False
    mixup: bool = False
    ruido: bool = False
    # SpecAugment
    n_mascaras_freq: int = 2
    largura_freq: int = 16
    n_mascaras_tempo: int = 2
    largura_tempo: int = 40
    # mixup
    alfa_mixup: float = 0.4
    # ruído
    prob_ruido: float = 0.5
    snr_min: float = 0.0
    snr_max: float = 20.0

    @property
    def nome(self) -> str:
        partes = [n for n, ativo in (("specaug", self.specaugment), ("mixup", self.mixup), ("ruido", self.ruido)) if ativo]
        return "+".join(partes) or "sem"


SEM = ConfigAumento()
CONFIGS_E2 = {
    "sem": SEM,
    "specaug": ConfigAumento(specaugment=True),
    "mixup": ConfigAumento(mixup=True),
    "ruido": ConfigAumento(ruido=True),
    "todas": ConfigAumento(specaugment=True, mixup=True, ruido=True),
}


# ----------------------------------------------------------------------------- ruído

def _sinal_de_5s(caminho, cfg: ConfigMel) -> np.ndarray:
    sinal = preproc.carregar_audio(caminho, cfg)
    n = cfg.amostras_janela
    if len(sinal) < n:
        sinal = np.tile(sinal, int(np.ceil(n / len(sinal))))
    return sinal[:n]


def carregar_ruidos(folds: list[int], cfg: ConfigMel = MEL128_32K) -> tuple[np.ndarray, pd.DataFrame]:
    """Formas de onda de 5 s dos ruídos da ESC-50 dos folds pedidos."""
    pasta = dados.pasta_dados() / "ruido"
    meta = pd.read_csv(pasta / "ruido.csv")
    meta = meta[meta["fold"].isin(folds)].reset_index(drop=True)
    sinais = np.stack([_sinal_de_5s(pasta / "audio" / f, cfg) for f in meta["filename"]])
    return sinais, meta


def preparar_ruido_treino(cfg: ConfigMel = MEL128_32K) -> None:
    """Log-Mel dos ruídos de treino (folds 1 a 4), em `R.npy` ao lado do `X.npy`."""
    sinais, meta = carregar_ruidos([1, 2, 3, 4], cfg)
    saida = preproc.pasta_config(cfg)
    np.save(saida / "R.npy", preproc.log_mel(sinais, cfg).astype(np.float16))
    meta.to_csv(saida / "ruido_treino.csv", index=False)


def misturar_onda(sinal: np.ndarray, ruido: np.ndarray, snr_db: float) -> np.ndarray:
    """Soma o ruído ao sinal com a SNR pedida, medida pela potência média das duas ondas."""
    p_sinal = np.mean(sinal ** 2) + 1e-12
    p_ruido = np.mean(ruido ** 2) + 1e-12
    ganho = np.sqrt(p_sinal / (p_ruido * 10 ** (snr_db / 10)))
    return sinal + ganho * ruido


def preparar_teste_ruidoso(cfg: ConfigMel = MEL128_32K, snrs=SNRS_TESTE, semente: int = dados.SEMENTE) -> None:
    """Versões do conjunto de teste com ruído do fold 5 da ESC-50, uma por SNR (E3).

    As janelas são as mesmas do `X.npy` (mesmo recorte determinístico) e cada janela recebe
    sempre o mesmo ruído em todas as SNRs, para que só a intensidade mude entre as versões.
    Gera `X_teste_snr<k>.npy` e `janelas_teste.csv`.
    """
    _, janelas = preproc.carregar(cfg)
    teste = janelas[janelas["particao"] == "teste"]
    ruidos, _ = carregar_ruidos([FOLD_RUIDO_TESTE], cfg)
    rng = np.random.default_rng(semente)
    pasta_audio = dados.pasta_dados() / "bruto" / "train_audio"

    por_snr = {snr: [] for snr in snrs}
    ordem = []
    for arquivo, grupo in tqdm(teste.groupby("filename", sort=False), desc="teste com ruído"):
        jan, inicios = preproc.janelas_da_gravacao(preproc.carregar_audio(pasta_audio / arquivo, cfg), cfg)
        assert np.allclose(inicios, grupo["inicio_s"].to_numpy(), atol=0.01), arquivo
        escolhidos = ruidos[rng.integers(len(ruidos), size=len(jan))]
        for snr in snrs:
            misturadas = np.stack([misturar_onda(s, r, snr) for s, r in zip(jan, escolhidos)])
            por_snr[snr].append(preproc.log_mel(misturadas, cfg).astype(np.float16))
        ordem.append(grupo)

    saida = preproc.pasta_config(cfg)
    pd.concat(ordem).to_csv(saida / "janelas_teste.csv", index=False)
    for snr in snrs:
        np.save(saida / f"X_teste_snr{snr}.npy", np.concatenate(por_snr[snr]))


def misturar_ruido_mel(x_db: torch.Tensor, r_db: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
    """Soma ruído a log-Mel em dB, no domínio de potência. Formas: (B, M, T), (B, M, T), (B,)."""
    p_x = torch.pow(10.0, x_db / 10)
    p_r = torch.pow(10.0, r_db / 10)
    ganho = p_x.mean(dim=(1, 2)) / (p_r.mean(dim=(1, 2)) * torch.pow(10.0, snr_db / 10))
    return 10 * torch.log10(p_x + ganho[:, None, None] * p_r + 1e-10)


# ----------------------------------------------------------------------------- lote

def spec_augment(x: torch.Tensor, cfg: ConfigAumento) -> torch.Tensor:
    """Máscaras de frequência e de tempo, com valor 0 (a média, depois da normalização)."""
    b, m, t = x.shape
    x = x.clone()
    dispositivo = x.device
    for _ in range(cfg.n_mascaras_freq):
        larg = torch.randint(0, cfg.largura_freq + 1, (b,), device=dispositivo)
        ini = (torch.rand(b, device=dispositivo) * (m - larg)).long()
        faixa = torch.arange(m, device=dispositivo)[None, :]
        mascara = (faixa >= ini[:, None]) & (faixa < (ini + larg)[:, None])
        x.masked_fill_(mascara[:, :, None], 0.0)
    for _ in range(cfg.n_mascaras_tempo):
        larg = torch.randint(0, cfg.largura_tempo + 1, (b,), device=dispositivo)
        ini = (torch.rand(b, device=dispositivo) * (t - larg)).long()
        faixa = torch.arange(t, device=dispositivo)[None, :]
        mascara = (faixa >= ini[:, None]) & (faixa < (ini + larg)[:, None])
        x.masked_fill_(mascara[:, None, :], 0.0)
    return x


def mixup(x: torch.Tensor, y: torch.Tensor, alfa: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Mistura cada exemplo com outro do lote; `y` são probabilidades (one-hot)."""
    lam = float(np.random.beta(alfa, alfa))
    perm = torch.randperm(x.shape[0], device=x.device)
    return lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]
