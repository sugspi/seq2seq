## RTE評価実験

1. `eval_gen.sh` を `ccg2lambda/en` ディレクトリに置く。
2. `bicond_eval.sh` `exp_entailment.sh` を `ccg2lambda` ディレクトリに置く。
3. `en_plain_<data>_<option>` 以下を `ccg2lambda` ディレクトリに置く。
4. ```./exp_entailment.sh <data> <ncores>``` を実行。 `<ncores>` はファイル分割数。
5. `gen_<data>` 以下に結果のファイルができる。Accuracyは`.score`ファイル、問題ごとのyes/no/unknown判定は`.table`ファイルを参照。

| data | SNLI  |   |
|:---|:---|:---|
|tamf| text | attention + masking + formula |
|taf| text | attention + formula |
|tmf| text | masking + formula |
|hamf| hypothesis | attention + masking + formula |
|haf| hypothesis | attention + formula |
|hmf| hypothesis | masking + formula |
|ts| hypothesis | Treebank Semantics (Butler 2016) |


