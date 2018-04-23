## RTE評価実験

1. `eval_gen.sh` を `ccg2lambda/en` ディレクトリに置く。
2. `bicond_eval.sh` `exp_entailment.sh` `extract_snli.sh`  を `ccg2lambda` ディレクトリに置く。
3. `plain` 以下の `taf.txt` 以下6つのファイルを `ccg2lambda` ディレクトリに置く。
4. ```./exp_entailment.sh <ncores>``` を実行。 `<ncores>` はファイル分割数。
5. `gen_{tamf,taf,tmf,hamf,haf,hmf}` 以下に結果のファイルができる。Accuracyは`.score`ファイル、問題ごとのyes/no/unknown判定は`.table`ファイルを参照。

|ファイル名 | SNLI  |   |
|:---|:---|:---|
|tamf| text | attention + masking + formula |
|taf| text | attention + formula |
|tmf| text | masking + formula |
|hamf| hypothesis | attention + masking + formula |
|haf| hypothesis | attention + formula |
|hmf| hypothesis | masking + formula |



