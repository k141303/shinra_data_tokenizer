import os
import re
import tqdm

from multiprocessing import Pool

from collections import Counter

try:
    from pyknp import Juman
except ModuleNotFoundError:
    raise ModuleNotFoundError("指定したトーカナイザーにはpyknpが必要です。\n$ pip install pyknp")

try:
    import mojimoji
except ModuleNotFoundError:
    raise ModuleNotFoundError("指定したトーカナイザーにはmojimojiが必要です。\n$ pip install mojimoji")

from data_utils import DataUtils, DataTools
from tokenization.vocab_utils import count_vocab
from tokenization.annotation_utils import annotation_mapper

def tokenize(inputs):
    temp_path, command, chunks = inputs

    if command == "jumanpp":
        parser = Juman()
    else:
        parser = Juman(jumanpp=False)

    errors = Counter()
    mapped_annotation = {}
    tokenized_documents = []
    for page_id, file_path, annotation in chunks:
        text = DataUtils.load_file(file_path)

        text = mojimoji.han_to_zen(text) # 全角に正規化(pyknpは半角記号に弱いため)

        tokenized_sentences = []
        for line in text.split("\n"):
            if len(line.strip()) == 0:
                tokenized_sentences.append([])
                continue

            try:
                parsed = parser.analysis(line)
                tokens = [mrph.midasi for mrph in parsed.mrph_list()]
            except: # 主に入力長の問題
                tokens = []

            if len(tokens) == 0: # 入力が長文すぎた場合のpyknpのエラーに対応
                try:
                    if "。" not in line: # 文ではない余計な文字列
                        for idx in range(0, len(line), 100): # 無理やり分割
                            parsed = parser.analysis(line[idx:idx + 100])
                            tokens += [mrph.midasi for mrph in parsed.mrph_list()]
                    else: # 分単位に区切って処理
                        for sentence in re.findall(".*?。", line):
                            parsed = parser.analysis(sentence)
                            tokens += [mrph.midasi for mrph in parsed.mrph_list()]
                except:
                    tokenized_sentences.append([])
                    errors["juman"] += 1
                    continue

            offsets = []
            if line != "".join(tokens): # 復元された文章が異なる場合(=>オフセットがずれている場合)
                branks = {}
                for m in re.finditer(r"[\xa0]|\s", line): # 消えた空白を把握
                    s, e = m.span(0)
                    branks[s] = m.group(0)

                # 消えた空白を補完しながらオフセットを計算
                temp = ""
                while len(temp) in branks: # 消えた空白の補完
                    temp += branks[len(temp)]
                for token in tokens:
                    offsets.append((
                        len(temp), len(temp)+len(token)
                    ))  #トークンの開始位置を保存
                    temp += token
                    while len(temp) in branks: # 消えた空白の補完
                        temp += branks[len(temp)]

                if temp != line:
                    errors["consistency"] += 1
            else: #　特に問題なく復元できる場合(=>オフセットがずれていない場合)
                #　オフセットを計算
                temp = ""
                for token in tokens:
                    offsets.append((
                        len(temp), len(temp)+len(token)
                    ))  #トークンの開始位置を保存
                    temp += token

            starts, ends = zip(*offsets)
            tokenized_sentences.append([*zip(tokens, starts, ends)])

        if annotation is not None:
            # アノテーションを各トークンにマップ
            mapped_annotation[page_id], match_errors = annotation_mapper(
                annotation,
                tokenized_sentences
            )
            errors["total_annotation"] += len(annotation)
            errors["match"] += match_errors

        tokenized_documents.append({"page_id":page_id, "tokens":tokenized_sentences})

    DataUtils.save_oneliner_json(temp_path, tokenized_documents, parallel=1)

    return errors, mapped_annotation, temp_path

def run_tokenize(args, shinra, command, num_jobs=100):
    temp_dir = os.path.join(args.output_dir, command, "_temp_files")
    os.makedirs(temp_dir, exist_ok=True)

    total_errors = Counter()
    t = tqdm.tqdm(total=len(shinra.categories)*num_jobs*3)
    for category in shinra.categories:
        # 書き出し先フォルダ
        output_dir = os.path.join(
            args.output_dir,
            command,
            shinra.to_c_cls(category),
            category
        )

        if os.path.exists(os.path.join(output_dir, "vocab.txt")):
            t.update(num_jobs*3)
            continue

        # 並列処理のためにジョブ分割
        targets = []
        for page_id, plain_path in shinra.plain_paths[category].items():
            targets.append((
                page_id,
                plain_path,
                shinra.annotations[category].get(page_id),
            ))
        jobs = DataTools.split_array(targets, num_jobs)
        jobs = [(os.path.join(temp_dir, f"{job_id}.json"), command, job) for job_id, job in enumerate(jobs)]

        # トークナイズ
        total_mapped_annotation, temp_paths = {}, []
        with Pool(args.parallel) as p:
            for errors, mapped_annotation, temp_path in p.imap_unordered(tokenize, jobs):
                total_errors += errors
                total_mapped_annotation.update(mapped_annotation)
                temp_paths.append(temp_path)
                t.update()
        assert len(total_mapped_annotation) == len(shinra.annotations[category]), \
            f"アノテーション数エラー\ntokenizer:mecab\nmecab_dic:{mecab_dic}\ncategory:{category}"

        #　語彙数カウント
        vocab = count_vocab(temp_paths, p_bar=t, parallel=args.parallel)

        # 書き出し先dir作成
        os.makedirs(os.path.join(output_dir, "tokens"), exist_ok=True)

        # 並列処理のためのジョブ作成
        jobs = []
        for temp_path in temp_paths:
            jobs.append((
                temp_path, vocab, output_dir
            ))

        # 書き出し
        with Pool(args.parallel) as p:
            for _ in p.imap_unordered(DataUtils.save_tokenized_file, jobs):
                t.update()

        DataUtils.save_oneliner_json(
            os.path.join(output_dir, f"{category}_dist.json"),
            DataTools.flatten(total_mapped_annotation.values()),
            parallel=args.parallel
        )
        DataUtils.save_file(
            os.path.join(output_dir, "vocab.txt"),
            "\n".join(vocab)
        )

    t.close()

    if total_errors["total_annotation"] != 0:
        print(f"Juman自体のエラー:{total_errors['juman']} \n"+
            f"トークンから復元された文章が異なる例:{total_errors['consistency']} \n"+
            f"トークンへのマッピングにより左右のいずれかがずれたアノテーション:{total_errors['match']/total_errors['total_annotation']}")

    return os.path.join(args.output_dir, command)
