# 森羅データセットトークナイザー

事前トークナイズ済みデータセットを作成するためのスクリプトです。  
以下の機能を持たせました。  

- トークナイズ
- 元のテキストのオフセットを各トークンに付与
- アノテーションのオフセットを対応するトークンへマッピング

現状対応しているトーカナイザーは以下の通りです。

- `mecab_ipadic`:MeCab IPA辞書 \※1
- `mecab_jumandic`:MeCab Juman辞書 [NICT BERT(BPE無し)](https://alaginrc.nict.go.jp/nict-bert/index.html)対応 \*1
- `Jumanpp`:Juman++ \*1\*2
- `kurohashi_bert`:Juman++ & BPE [黒橋研BERT](http://nlp.ist.i.kyoto-u.ac.jp/index.php?BERT%E6%97%A5%E6%9C%AC%E8%AA%9EPretrained%E3%83%A2%E3%83%87%E3%83%AB)対応版 \*1

\※1:半角を全角に正規化  
\*2:公開データは`Version: 2.0.0-rc2 / Dictionary: 20180202-2cca748 / LM: K:20180217-6c28641 L:20180221-fd8a4b63 F:20171214-9d125cb`

## 実行方法

以下の様にデータセットごとに指定してトークナイズできます。

~~~
# データセットのパスを環境変数に格納(処理したいデータセットのみ格納してください)
export SHINRA2020EVENT=[Eventデータセットのパス]
export SHINRA2020FACILITY=[Facilityデータセットのパス]
export SHINRA2020JP5=[JP-5データセットのパス]
export SHINRA2020LOCATION=[Locationデータセットのパス]
export SHINRA2020ORGANIZATION=[Organizationデータセットのパス]

export MECAB_IPADIC_DIR='ipadicへのパス(mecab_ipadicの場合に必要)'
export MECAB_JUMANDIC_DIR='jumandicへのパス(mecab_jumandicの場合に必要)'

python3 code/main.py \
    --tokenizers mecab_ipadic mecab_jumandic juman \
    --output_dir ./outputs \
    --parallel -1 # 並列コア数(-1の場合は最大)
~~~

使用したいトークナイザーを`--tokenizer`に指定してください。

## 出力ファイルの見方

例えば、`mecab_ipadic`で`JP-5`を処理した場合、`Airport`カテゴリーは以下の様に書き出されます。  

~~~
./outputs/mecab_ipadic/JP-5/Airport/
 ├ Airport_dist.json
 ├ vocab.txt
 └ tokens/
    ├ 1001711.txt
    ├ 175701.txt
    ≈
    └ 999854.txt
~~~

### ・(カテゴリー名)_dist.json

~~~
{"page_id": "3507880", "title": "アンパーラ空港", "attribute": "別名", "html_offset": {"start": {"line_id": 39, "offset": 395}, "end": {"line_id": 39, "offset": 406}, "text": "SLAF Ampara"}, "text_offset": {"start": {"line_id": 39, "offset": 61}, "end": {"line_id": 39, "offset": 72}, "text": "SLAF Ampara"}, "ENE": "1.6.5.3", "token_offset": {"start": {"line_id": 39, "offset": 7}, "end": {"line_id": 39, "offset": 9}, "text": "SLAF Ampara"}}
{"page_id": "3507880", "title": "アンパーラ空港", "attribute": "別名", "html_offset": {"start": {"line_id": 70, "offset": 375}, "end": {"line_id": 70, "offset": 389}, "text": "Ampara Airport"}, "text_offset": {"start": {"line_id": 70, "offset": 76}, "end": {"line_id": 70, "offset": 90}, "text": "Ampara Airport"}, "ENE": "1.6.5.3", "token_offset": {"start": {"line_id": 70, "offset": 23}, "end": {"line_id": 70, "offset": 25}, "text": "Ampara Airport"}}
...
~~~

配布されている学習データと同じ形式です。  
`token_offset`->`offset`の値が各トークンの(開始|終了)インデックスになっています。

### ・vocab.txt

~~~
.
","
の
、
空港
月
":
":"
~~~

語彙ファイルです。  
主にデータサイズを圧縮するために使用しています。  
カテゴリーごとに集計されています。
`kurohashi_bert`では[黒橋研BERT](http://nlp.ist.i.kyoto-u.ac.jp/index.php?BERT%E6%97%A5%E6%9C%AC%E8%AA%9EPretrained%E3%83%A2%E3%83%87%E3%83%AB)の語彙ファイルを代わりに使用しています。

>note
>
>語彙ファイルの改行記号は`\n`です。
>Pythonの場合は`split("\n")`で分割してください。

### ・(page_id).txt

```
717,0,2 11,3,4 580,4,8 20,8,9 23,9,10 240,10,12 510,12,13
32,0,1 30,1,2 3812,2,5 30,5,6 5056,6,9 245,9,10 494,10,12 832,12,14
1117,0,2 30,2,3 5207,3,6 657,6,7
148,0,2 8,2,3 2539,4,10 2104,11,17 3292,18,28 717,28,30 22,30,31 671,
31,34 1748,34,37 496,37,39 57,39,40 24,40,41 119,41,42 240,42,44 510,
44,45 22,45,46 3565,46,48 3566,48,53 33,53,55 4,55,57 2749,57,59 496,59,61 57,61,62 32,62,63 119,63,64 473,64,67
```

各ファイルをトークナイズしたデータです。  
(語彙id, 開始オフセット, 終了オフセット)を`,`で結合したものを、半角スペースで結合してあります。  
語彙idは`vocab.txt`の語彙と対応しています(語彙idは0始まりです。つまり`717`の場合`vocab.txt`の`718`行目の語彙を示しています。)。  
開始オフセット、終了オフセットはそれぞれ元のテキストでのオフセットを示しています。  
各行は元のテキストの行と対応していますので、`line_id`は元データと同様に取得できます。

## 補足等

学習データでのオフセットが必ずしもトークンの境目と一致するとは限りません。
そのため、トークンの間にオフセットが存在した場合は、そのトークンも含む様に`token_offset`->`offset`が決定されます。
(つまりプレーンテキスト上でより長い範囲が選択される様決定されます。)

アノテーションの左右もしくは両方がトークンへのマッピングによりずれた割合は次の様になります。

|トークナイザー|ズレ率|
|:---|---:|
|MeCab IPA辞書|1.21%|
