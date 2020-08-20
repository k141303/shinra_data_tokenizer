from collections import Counter

from multiprocessing import Pool
import multiprocessing as multi

from data_utils import DataUtils

def count_vocab_func(file_path):
    # 並列処理用の関数
    tokenized_documents = DataUtils.load_oneliner_json(file_path, parallel=1)

    total_tokens = []
    for d in tokenized_documents:
        for tokens in d["tokens"]:
            if len(tokens) == 0:
                continue
            tokens, *_ = map(list, zip(*tokens))
            total_tokens += tokens
    return Counter(total_tokens)

def count_vocab(file_paths, p_bar=None, parallel=multi.cpu_count()):
    #　語彙数カウント
    total_vocab = Counter()
    with Pool(parallel) as p:
        for vocab in p.imap_unordered(count_vocab_func, file_paths):
            total_vocab += vocab
            if p_bar is not None:
                p_bar.update()
    total_vocab = [token for token, cnt in total_vocab.most_common()]
    return total_vocab
