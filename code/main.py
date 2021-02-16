import argparse
import multiprocessing as multi
import os

from data_utils import ShinraDataset, TokenizedDataset
from tokenization.annotation_utils import annotation_checker

# データセット用の環境変数リスト
DATASET_ENV = {
    "JP-5": "SHINRA2020JP5",
    "Organization": "SHINRA2020ORGANIZATION",
    "Location": "SHINRA2020LOCATION",
    "Event": "SHINRA2020EVENT",
    "Facility": "SHINRA2020FACILITY",
}

# データセット用の環境変数からパスを取得
DATASET_DIR = {
    c_cls: os.environ.get(env)
    for c_cls, env in DATASET_ENV.items()
    if os.environ.get(env) is not None
}
assert (
    len(DATASET_DIR) != 0
), f"次の環境変数の1つ以上にデータセットのパスを格納する必要があります。\n{list(DATASET_ENV.values())}"


def load_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizers",
        nargs="+",
        choices=[
            "mecab_ipadic",
            "mecab_jumandic",
            "jumanpp",
            "tohoku_bert_mecab_ipadic_bpe",
        ],
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="対象カテゴリーを制限できます。指定がない場合は全てのカテゴリーが処理されます。",
    )
    parser.add_argument(
        "--parallel",
        default=1,
        type=int,
        choices=[*range(1, multi.cpu_count() + 1)] + [-1],
        help="1(default)~コア数で並列数を指定できます。-1の場合は最大コア数が使用されます。",
    )
    parser.add_argument("--output_dir", default="./outputs", help="出力先フォルダ")
    return parser.parse_args()


if __name__ == "__main__":
    args = load_arg()

    if args.parallel == -1:
        args.parallel = multi.cpu_count()

    shinra = ShinraDataset(args, DATASET_DIR)

    if "mecab_ipadic" in args.tokenizers:
        from tokenization import mecab

        mecab_ipadic_output_dir = mecab.run_tokenize(args, shinra, "ipadic")

    if "mecab_jumandic" in args.tokenizers:
        from tokenization import mecab

        mecab_juman_output_dir = mecab.run_tokenize(args, shinra, "jumandic")

    if "jumanpp" in args.tokenizers:
        from tokenization import juman

        jumanpp_output_dir = juman.run_tokenize(args, shinra, command="jumanpp")

    if "tohoku_bert_mecab_ipadic_bpe" in args.tokenizers:
        from tokenization import tohoku_bert

        tohoku_bert.run_tokenize(args, shinra)
