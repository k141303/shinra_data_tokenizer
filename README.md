# shinra_data_tokenizer

事前トークナイズ済みデータセットを作成するためのスクリプトです。  
以下の機能を持たせました。  

- トークナイズ
- 元のテキスト中のオフセットをトークンに付与
- アノテーションのオフセットを対応するトークンへマッピング

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
{"page_id": "3407002", "title": "ラジャ・ハジ・フィサビッリラー空港", "attribute": "別名", "ENE": "1.6.5.3", "token_offset": {"start": {"line_id": 67, "offset": 14}, "end": {"line_id": 67, "offset": 18}, "text": "Raja Haji Fisabilillah Airport"}}
{"page_id": "3407002", "title": "ラジャ・ハジ・フィサビッリラー空港", "attribute": "別名", "ENE": "1.6.5.3", "token_offset": {"start": {"line_id": 34, "offset": 8}, "end": {"line_id": 34, "offset": 13}, "text": "Raja Haji Fisabilillah International Airport"}}
...
~~~

配布されている学習データと同じ形式です。  
`offset`の値が各トークンの(開始|終了)インデックスになっています。

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
カテゴリーごとに集計されていることに注意してください。

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
