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
