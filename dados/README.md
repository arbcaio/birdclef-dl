# dados/

Tudo nesta pasta, exceto este arquivo, fica fora do git. O repositório é público e as regras de
competição do Kaggle não permitem redistribuir os dados.

Layout esperado (o mesmo no Drive compartilhado e na máquina local):

```
dados/
  bruto/          # download do BirdCLEF: train_audio/, train_metadata.csv, ...
  ruido/          # ESC-50 (chuva, vento, insetos) para mistura de ruído
  subconjunto/    # lista das espécies e gravações escolhidas (CSV gerado pela EDA)
  espectrogramas/ # log-Mel pré-computados, uma pasta por configuração
    mel128_32k/   # M1 e M2
    mel64_32k/    # M3 (PANNs)
    mel128_16k/   # M4 (AST)
  divisoes/       # teste fixo e folds agrupados por gravação
```

O CSV do subconjunto e o das divisões são pequenos e definem os experimentos; se o grupo quiser
versioná-los, mova-os para `resultados/` em vez de abrir exceção aqui.
