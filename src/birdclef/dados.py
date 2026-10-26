"""Metadados, escolha do subconjunto e divisões de treino e teste.

O subconjunto e as divisões são feitos por gravação: cada linha é uma gravação, e as janelas de
5 s geradas depois herdam a partição e o fold da gravação de onde vieram. Assim janelas da mesma
gravação nunca caem em partições diferentes.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import StratifiedKFold, train_test_split

SEMENTE = 42


def pasta_dados() -> Path:
    """Pasta `dados/` na raiz do repositório (no Colab, o link para o Drive)."""
    return Path(__file__).resolve().parents[2] / "dados"


def carregar_metadados(pasta_bruto: Path | None = None) -> pd.DataFrame:
    """`train.csv` com a classe taxonômica (`class_name`) vinda do `taxonomy.csv`."""
    pasta_bruto = pasta_bruto or pasta_dados() / "bruto"
    meta = pd.read_csv(pasta_bruto / "train.csv")
    taxonomia = pd.read_csv(pasta_bruto / "taxonomy.csv")
    return meta.merge(taxonomia[["primary_label", "class_name"]], on="primary_label", how="left")


def duracoes(meta: pd.DataFrame, pasta_audio: Path | None = None) -> pd.Series:
    """Duração em segundos de cada gravação, lida do cabeçalho do .ogg (sem decodificar o áudio)."""
    pasta_audio = pasta_audio or pasta_dados() / "bruto" / "train_audio"
    return meta["filename"].map(lambda f: sf.info(pasta_audio / f).duration).rename("duracao_s")


def selecionar_subconjunto(
    meta: pd.DataFrame,
    n_especies: int = 20,
    max_por_especie: int = 150,
    rating_min: float = 4.0,
    semente: int = SEMENTE,
) -> pd.DataFrame:
    """Escolhe as espécies e as gravações do projeto.

    Só aves, só gravações do xeno-canto (as únicas com nota de qualidade) com `rating` mínimo.
    Ficam as `n_especies` espécies com mais gravações que passam no filtro e, de cada uma, até
    `max_por_especie` gravações sorteadas com semente fixa.
    """
    boas = meta[
        (meta["class_name"] == "Aves")
        & (meta["collection"] == "XC")
        & (meta["rating"] >= rating_min)
    ]
    contagem = boas["primary_label"].value_counts()
    especies = contagem.index[:n_especies]
    escolhidas = boas[boas["primary_label"].isin(especies)]
    sub = (
        escolhidas.sample(frac=1, random_state=semente)
        .groupby("primary_label")
        .head(max_por_especie)
        .sort_values(["primary_label", "filename"])
        .reset_index(drop=True)
    )
    return sub


def criar_divisoes(
    sub: pd.DataFrame, frac_teste: float = 0.2, k: int = 5, semente: int = SEMENTE
) -> pd.DataFrame:
    """Teste fixo estratificado por espécie e `k` folds estratificados no restante.

    Devolve `sub` com as colunas `particao` ("teste" ou "treino") e `fold` (0 a k-1 no treino,
    -1 no teste). O teste só é usado na avaliação final.
    """
    sub = sub.reset_index(drop=True).copy()
    idx_treino, idx_teste = train_test_split(
        sub.index, test_size=frac_teste, stratify=sub["primary_label"], random_state=semente
    )
    sub["particao"] = "treino"
    sub.loc[idx_teste, "particao"] = "teste"
    sub["fold"] = -1
    treino = sub.loc[idx_treino]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=semente)
    for fold, (_, idx_val) in enumerate(skf.split(treino, treino["primary_label"])):
        sub.loc[treino.index[idx_val], "fold"] = fold
    return sub


def resumo_divisoes(div: pd.DataFrame) -> pd.DataFrame:
    """Gravações por espécie em cada fold e no teste, para conferir a estratificação."""
    coluna = np.where(div["particao"] == "teste", "teste", "fold " + div["fold"].astype(str))
    return pd.crosstab(div["primary_label"], coluna, margins=True, margins_name="total")
