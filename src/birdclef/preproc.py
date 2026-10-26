"""Pré-processamento: janelas de 5 s e espectrogramas log-Mel pré-computados.

Cada gravação vira até `max_janelas` janelas de 5 s com 50% de sobreposição. Gravações com menos
de 5 s são completadas repetindo o próprio sinal (em vez de silêncio). Janelas muito mais fracas
que a janela mais forte da mesma gravação são descartadas, porque o rótulo vale para a gravação
inteira e trechos sem canto viram rótulos errados (rótulo fraco). Quando sobram janelas demais
(gravações longas), ficam `max_janelas` igualmente espaçadas.

Os espectrogramas ficam em dB com referência fixa (`ref=1`), sem normalizar por janela, para que
o ruído possa ser somado depois no domínio de potência (`aumento.misturar_ruido_mel`).

Saída em `dados/espectrogramas/<config>/`: `X.npy` (N, n_mels, quadros) em float16 e
`janelas.csv` com a gravação, a espécie, a partição, o fold e o início de cada janela.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import librosa
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from birdclef import dados


@dataclass(frozen=True)
class ConfigMel:
    nome: str = "mel128_32k"
    sr: int = 32000
    n_mels: int = 128
    fmin: float = 50.0
    fmax: float = 14000.0
    n_fft: int = 2048
    hop: int = 512
    duracao_janela: float = 5.0
    passo_janela: float = 2.5  # 50% de sobreposição
    max_janelas: int = 12
    limiar_db: float = 20.0  # descarta janelas X dB abaixo da mais forte da gravação

    @property
    def amostras_janela(self) -> int:
        return int(self.sr * self.duracao_janela)

    @property
    def quadros(self) -> int:
        return 1 + self.amostras_janela // self.hop


# M1 e M2. As configurações do PANNs (64 bandas a 32 kHz) e do AST (128 bandas a 16 kHz) são da
# Parte 2 e usam a mesma função com outro ConfigMel.
MEL128_32K = ConfigMel()


def log_mel(sinal: np.ndarray, cfg: ConfigMel) -> np.ndarray:
    """Espectrograma log-Mel em dB (ref=1) de um ou mais sinais de `amostras_janela` amostras."""
    potencia = librosa.feature.melspectrogram(
        y=sinal, sr=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop, n_mels=cfg.n_mels,
        fmin=cfg.fmin, fmax=cfg.fmax, power=2.0,
    )
    return librosa.power_to_db(potencia, ref=1.0, amin=1e-10, top_db=None)


def carregar_audio(caminho: Path, cfg: ConfigMel) -> np.ndarray:
    sinal, _ = librosa.load(caminho, sr=cfg.sr, mono=True)
    return sinal


def janelas_da_gravacao(sinal: np.ndarray, cfg: ConfigMel) -> tuple[np.ndarray, np.ndarray]:
    """Recorta, filtra por energia e limita as janelas de uma gravação.

    Devolve as janelas (n, amostras) e o início de cada uma em segundos.
    """
    n = cfg.amostras_janela
    if len(sinal) < n:
        sinal = np.tile(sinal, int(np.ceil(n / max(len(sinal), 1))))[:n]
    passo = int(cfg.sr * cfg.passo_janela)
    inicios = np.arange(0, len(sinal) - n + 1, passo)
    janelas = np.stack([sinal[i:i + n] for i in inicios])

    energia_db = 10 * np.log10(np.mean(janelas ** 2, axis=1) + 1e-12)
    manter = energia_db >= energia_db.max() - cfg.limiar_db
    janelas, inicios = janelas[manter], inicios[manter]

    if len(janelas) > cfg.max_janelas:
        idx = np.linspace(0, len(janelas) - 1, cfg.max_janelas).round().astype(int)
        janelas, inicios = janelas[idx], inicios[idx]
    return janelas, inicios / cfg.sr


def pasta_config(cfg: ConfigMel) -> Path:
    return dados.pasta_dados() / "espectrogramas" / cfg.nome


def preprocessar(div: pd.DataFrame, cfg: ConfigMel = MEL128_32K, pasta_audio: Path | None = None) -> Path:
    """Gera `X.npy` e `janelas.csv` para todas as gravações de `div` (saída de `criar_divisoes`)."""
    pasta_audio = pasta_audio or dados.pasta_dados() / "bruto" / "train_audio"
    saida = pasta_config(cfg)
    saida.mkdir(parents=True, exist_ok=True)

    blocos, linhas = [], []
    for g in tqdm(div.itertuples(index=False), total=len(div), desc=f"log-Mel {cfg.nome}"):
        janelas, inicios = janelas_da_gravacao(carregar_audio(pasta_audio / g.filename, cfg), cfg)
        blocos.append(log_mel(janelas, cfg).astype(np.float16))
        for ini in inicios:
            linhas.append((g.filename, g.primary_label, g.particao, g.fold, round(float(ini), 2)))

    X = np.concatenate(blocos)
    np.save(saida / "X.npy", X)
    janelas = pd.DataFrame(linhas, columns=["filename", "primary_label", "particao", "fold", "inicio_s"])
    janelas.to_csv(saida / "janelas.csv", index=False)
    (saida / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    return saida


def carregar(cfg: ConfigMel = MEL128_32K, sufixo: str = "") -> tuple[np.ndarray, pd.DataFrame]:
    """`X` (mapeado em memória) e `janelas.csv` de uma configuração já pré-processada."""
    pasta = pasta_config(cfg)
    X = np.load(pasta / f"X{sufixo}.npy", mmap_mode="r")
    janelas = pd.read_csv(pasta / f"janelas{sufixo}.csv")
    return X, janelas
