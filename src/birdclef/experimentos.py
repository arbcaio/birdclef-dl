"""Experimentos da Parte 1 com o M1 (CNN do zero).

- E2 (ablação do aumento): cada configuração de `aumento.CONFIGS_E2`, 3 sementes, fold 0 de
  validação; avaliada na validação e no teste limpo.
- E3 (robustez a ruído): os modelos "sem" e "todas" do E2, avaliados no teste limpo e com ruído
  do fold 5 da ESC-50 a 20, 10, 5 e 0 dB.

Uso: `python -m birdclef.experimentos [e2|e3|tudo]`. Cada execução de treino é guardada em
`checkpoints/` e as métricas em `resultados/`, e execuções já feitas são puladas.
"""

from dataclasses import asdict
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch

from birdclef import aumento, modelos, preproc, treino
from birdclef.treino import ConfigTreino, Resultado

RAIZ = Path(__file__).resolve().parents[2]
PASTA_RESULTADOS = RAIZ / "resultados"
PASTA_CHECKPOINTS = RAIZ / "checkpoints"
SEMENTES = (42, 43, 44)
FOLD_VAL = 0


def carregar_dados(cfg_mel=preproc.MEL128_32K):
    """`X` em memória, índice das janelas e log-Mel dos ruídos de treino."""
    X, janelas = preproc.carregar(cfg_mel)
    R = np.load(preproc.pasta_config(cfg_mel) / "R.npy")
    return np.ascontiguousarray(X), janelas, R


def dados_teste(X, janelas):
    idx = np.flatnonzero((janelas["particao"] == "teste").to_numpy())
    return X[idx], janelas.iloc[idx].reset_index(drop=True)


def nome_execucao(modelo: str, aum: str, semente: int, fold: int = FOLD_VAL) -> str:
    return f"{modelo}_{aum}_s{semente}_f{fold}"


def carregar_checkpoint(caminho: Path, modelo: str = "M1") -> Resultado:
    ck = torch.load(caminho, map_location=treino.dispositivo(), weights_only=False)
    m = modelos.criar(modelo, len(ck["classes"])).to(treino.dispositivo())
    m.load_state_dict(ck["estado"])
    return Resultado(m, ck["classes"], tuple(ck["norm"]), pd.DataFrame(ck["historico"]), ck["metricas_val"], ck["segundos"])


def e2(X, janelas, R, sementes=SEMENTES, cfg=ConfigTreino()) -> pd.DataFrame:
    X_te, jan_te = dados_teste(X, janelas)
    saida = PASTA_RESULTADOS / "e2"
    saida.mkdir(parents=True, exist_ok=True)
    linhas = []
    for aum_nome, cfg_aum in aumento.CONFIGS_E2.items():
        for semente in sementes:
            nome = nome_execucao(cfg.modelo, aum_nome, semente)
            ck = PASTA_CHECKPOINTS / f"{nome}.pt"
            if ck.exists():
                res = carregar_checkpoint(ck, cfg.modelo)
            else:
                print(f"--- {nome}", flush=True)
                c = ConfigTreino(**{**asdict(cfg), "semente": semente})
                res = treino.treinar(X, janelas, FOLD_VAL, cfg_aum, c, R=R)
                treino.salvar(res, ck, historico=res.historico.to_dict("list"),
                              metricas_val=res.metricas_val, segundos=res.segundos, aumento=asdict(cfg_aum))
                res.historico.to_csv(saida / f"historico_{nome}.csv", index=False)
            m_te, y, probs = treino.avaliar_teste(res, X_te, jan_te)
            np.save(saida / f"probs_teste_{nome}.npy", probs)
            linhas.append({
                "aumento": aum_nome, "semente": semente, "epocas": len(res.historico),
                "melhor_epoca": int(res.metricas_val["epoca"]), "minutos": res.segundos / 60,
                **{f"val_{k}": res.metricas_val[k] for k in ("f1_macro", "acc_top1", "acc_top3", "map")},
                **{f"teste_{k}": v for k, v in m_te.items()},
            })
            print(linhas[-1], flush=True)
    df = pd.DataFrame(linhas)
    df.to_csv(saida / "e2_execucoes.csv", index=False)
    resumo = df.drop(columns="semente").groupby("aumento", sort=False).agg(["mean", "std"])
    resumo.to_csv(saida / "e2_resumo.csv")
    np.save(saida / "y_teste.npy", y)
    (saida / "classes.json").write_text(json.dumps(res.classes))
    return df


def e3(janelas, sementes=SEMENTES, modelo="M1", configs=("sem", "todas")) -> pd.DataFrame:
    cfg_mel = preproc.MEL128_32K
    pasta = preproc.pasta_config(cfg_mel)
    jan_te = pd.read_csv(pasta / "janelas_teste.csv")
    X_limpo, jan_limpo = dados_teste(np.load(pasta / "X.npy", mmap_mode="r"), janelas)
    assert (jan_limpo[["filename", "inicio_s"]].to_numpy() == jan_te[["filename", "inicio_s"]].to_numpy()).all()
    versoes = {"limpo": np.asarray(X_limpo)}
    for snr in aumento.SNRS_TESTE:
        versoes[f"{snr} dB"] = np.load(pasta / f"X_teste_snr{snr}.npy")

    linhas = []
    for aum_nome in configs:
        for semente in sementes:
            res = carregar_checkpoint(PASTA_CHECKPOINTS / f"{nome_execucao(modelo, aum_nome, semente)}.pt", modelo)
            for versao, Xv in versoes.items():
                m, _, _ = treino.avaliar_teste(res, Xv, jan_te)
                linhas.append({"aumento": aum_nome, "semente": semente, "teste": versao, **m})
    df = pd.DataFrame(linhas)
    saida = PASTA_RESULTADOS / "e3"
    saida.mkdir(parents=True, exist_ok=True)
    df.to_csv(saida / "e3_execucoes.csv", index=False)
    df.drop(columns="semente").groupby(["aumento", "teste"], sort=False).agg(["mean", "std"]).to_csv(saida / "e3_resumo.csv")
    return df


if __name__ == "__main__":
    qual = sys.argv[1] if len(sys.argv) > 1 else "tudo"
    X, janelas, R = carregar_dados()
    print(f"X {X.shape}, {modelos.n_parametros(modelos.criar('M1', janelas['primary_label'].nunique())):,} parâmetros", flush=True)
    if qual in ("e2", "tudo"):
        e2(X, janelas, R)
    if qual in ("e3", "tudo"):
        e3(janelas)
