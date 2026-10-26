# dados/

Tudo nesta pasta, exceto este arquivo, fica fora do git. O repositório é público e as regras de
competição do Kaggle não permitem redistribuir os dados.

Layout esperado (o mesmo no Drive compartilhado e na máquina local):

```
dados/
  bruto/          # BirdCLEF 2025: train.csv, taxonomy.csv e train_audio/ (sem train_soundscapes/)
  ruido/          # ESC-50 (chuva, vento, insetos) para mistura de ruído
  subconjunto/    # divisoes.csv: gravações escolhidas, partição e fold (notebooks/01_eda)
  espectrogramas/ # log-Mel pré-computados, uma pasta por configuração
    mel128_32k/   # M1 e M2
    mel64_32k/    # M3 (PANNs)
    mel128_16k/   # M4 (AST)
```

O `divisoes.csv` é gerado de forma determinística (semente 42) pelo `notebooks/01_eda.ipynb`, a
partir do `train.csv`; por isso não precisa ir para o git. Quem tiver os dados brutos o regenera
idêntico.
