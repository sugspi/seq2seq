
  make_data.ipynb:{snli|京大含意関係認識データセット|京大リードコーパス}のデータからテキスト取り出し+candcパーザーからsuccessを取り出し,変数名を統一化しjsonに保存

el_dup.pyはjosn形式だった教師データを#で配列構造にしたもの ...(1)
el_dup_jp.pyは日本語の形態素解析をしたデータを作る機能（tubameでmecabが使えないため）...(1)
nltk2pn : Pを除去 ...(2)

nltk2token.py :token区切りでデータ化...(3)
nltk2tree.py :木構造化 ......(3)
nltk2pregraph.py ：networkxよりグラフ化 ...(3)

(1),(2)は必ず実行してから(3)をそれぞれやる


メモ
* sort text | uniq (| wc -l) でパーズする前にユニークな文か確認する必要がある場合もある
京大リードコーパス作成の時に使ったコマンド
sort plain.txt | uniq -c | sort -nr | sed 's/^\s*[0-9]\s//g' > uniq_plain.txt
cat disc.txt | grep -v '^#' | grep -v '^\s*$' | grep -v '^[0-9]-[0-9]' | sed 's/^[0-9] //g' | tr -d '\n' | sed 's/。/。\n/g' > sentences.txt
変な記号をとったversion
cat sentences.txt | sed 's/^・//g' | sed 's/^●//g' | sed 's/^◆//g' | sed 's/^。$//g' | sed '/^\s*$/d' > sentences.clean.txt
