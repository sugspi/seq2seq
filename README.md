# seq2seq


## src ...embedding別の実験用ソース
(1)char 記号単位<br>
(2)word トークン<br>
(3)tree ポーランド記法に変化するnltk2pn.py <br>
(4)graph グラフはdata/のnltk2graph.pyであらかじめ作らなくてはならない<br>
(5)bleu_analysis.ipynb bleuの出力結果を解析する



## kyotoU .... 京大含意関係コーパスから作ったデータ
ku_extract.ipynbで一行単位で取り出し&semから取り出し<br>


## snli   .... SNLI devからつくったデータ <br>
make_data.ipynbで一行取り出し，semから取り出し

el_dup.py        
el_dup_jp.py   
#区切りにするためのファイル

nltk2pn.py　ポーランド記法に変換

logic_parser.py
nltk2pregraph.py snli
グラフに変換
