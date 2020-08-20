from multiprocessing import Pool

from data_utils import DataTools, DataUtils

def annotation_mapper(annotation, tokenized_sentences, ann_key="token"):
    """
    オフセットを各トークンにマップします。
    """
    match_errors = 0
    for ann in annotation:
        if ann.get("text_offset") is None: # 総称フラグ関連
            continue

        start, end = ann["text_offset"]["start"], ann["text_offset"]["end"]

        ann["token_offset"] = {
            "start":{"line_id":start["line_id"]},
            "end":{"line_id":end["line_id"]},
            "text":ann["text_offset"]["text"]
        }

        match_error = False

        tokens = tokenized_sentences[start["line_id"]]
        for i, (token, s, e) in enumerate(tokens):
            if start["offset"] >= s and start["offset"] < e:
                ann["token_offset"]["start"]["offset"] = i
                if s != start["offset"]:
                    match_error = True
                break

        tokens = tokenized_sentences[end["line_id"]]
        for i, (token, s, e) in enumerate(tokens):
            if end["offset"] > s and end["offset"] <= e:
                ann["token_offset"]["end"]["offset"] = i + 1
                if e != end["offset"]:
                    match_error = True
                break

        assert ann["token_offset"]["start"].get("offset") is not None, \
            f"Startオフセットマッチエラー\ntokens:{tokens}\nann:{ann['text_offset']}"
        assert ann["token_offset"]["end"].get("offset") is not None, \
            f"Endオフセットマッチエラー\ntokens:{tokens}\nann:{ann['text_offset']}"

        del ann["text_offset"]
        del ann["html_offset"]

        match_errors += match_error

    return annotation, match_errors

def check_annotation(inputs):
    for page_id, plain_path, tokens_path, offsets_paths, annotation, vocab in inputs:
        sencence_token_ids = DataUtils.load_file(tokens_path).splitlines()
        sentence_offsets = DataUtils.load_file(offsets_paths).splitlines()
        for ann in annotation:
            start, end = ann["token_offset"]["start"], ann["token_offset"]["end"]

            token_ids = sencence_token_ids[start["line_id"]].split()
            tokens = [vocab[int(token_id)] for token_id in token_ids]
            offsets = sentence_offsets[start["line_id"]].split()
            offsets = [*map(int, offsets)]

            assert False, end_offsets



def annotation_checker(args, shinra, tokenized_shinra, num_jobs=100):
    for category in tokenized_shinra.categories:
        targets = []
        for page_id in tokenized_shinra.annotations[category]:
            targets.append((
                page_id,
                shinra.plain_paths[category][page_id],
                tokenized_shinra.tokens_paths[category][page_id],
                tokenized_shinra.offsets_paths[category][page_id],
                tokenized_shinra.annotations[category][page_id],
                tokenized_shinra.vocab[category]
            ))

        jobs = DataTools.split_array(targets, num_jobs)
        with Pool(1) as p:
            for _ in p.imap_unordered(check_annotation, jobs):
                pass
