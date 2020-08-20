import os
import re
import json
import glob
import tqdm

from collections import defaultdict

from multiprocessing import Pool
import multiprocessing as multi

from html.parser import HTMLParser

class DataUtils(object):
    @staticmethod
    def load_oneliner_json(file_path, parallel=multi.cpu_count()):
        with open(file_path, "r") as f:
            if parallel == 1:
                return [*map(json.loads, f)]
            with Pool(parallel) as p:
                return p.map(json.loads, f)

    @staticmethod
    def load_annotation(category, file_path):
        data = defaultdict(list)
        with open(file_path, "r") as f:
            for d in map(json.loads, f):
                data[str(d["page_id"])].append(d)
        return category, data

    @staticmethod
    def json_dumps(d):
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def save_oneliner_json(cls, file_path, data, parallel=multi.cpu_count()):
        if parallel == 1:
            dumps = map(cls.json_dumps, data)
        else:
            with Pool(parallel) as p:
                dumps = p.map(cls.json_dumps, data)
        with open(file_path, "w") as f:
            f.write("\n".join(dumps))

    @classmethod
    def wrp_load_annotation(cls, inputs):
        return cls.load_annotation(*inputs)

    @staticmethod
    def load_file(file_path):
        with open(file_path, "r") as f:
            return f.read()

    @staticmethod
    def save_file(file_path, data):
        with open(file_path, "w") as f:
            f.write(data)

    @classmethod
    def save_tokenized_file(cls, inputs):
        temp_path, vocab, output_dir = inputs

        tokenized_documents = cls.load_oneliner_json(temp_path, parallel=1)
        for d in tokenized_documents:
            sentence_token_ids = []
            for tokens in d["tokens"]:
                if len(tokens) == 0:
                    sentence_token_ids.append("")
                    continue

                token_ids = [",".join(map(str, [vocab.index(token), s, e])) for (token, s, e) in tokens]
                sentence_token_ids.append(" ".join(token_ids))

            DataUtils.save_file(
                os.path.join(output_dir, "tokens", f"{d['page_id']}.txt"),
                "\n".join(sentence_token_ids)
            )

class DataTools(object):
    @staticmethod
    # リストをn分割
    def split_array(array, n):
        for i in range(n):
            yield array[i * len(array) // n:(i + 1) * len(array) // n]

    @staticmethod
    def flatten(arrays):
        new_array = []
        for array in arrays:
            new_array += array
        return new_array

class IgnoreSentenceParser(HTMLParser):
    def __init__(self):
        self.start_line = None
        self.ignore_lines = set()
        super().__init__()

    def reset(self):
        self.start_line = None
        self.ignore_lines = set()
        super().reset()

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            start_line, _ = self.getpos()
            self.start_line = start_line - 1

    def handle_endtag(self, tag):
        if tag == "script" and self.start_line is not None:
            end_line, _ = self.getpos()
            self.ignore_lines |= set(range(self.start_line, end_line))
            self.start_line = None

class ShinraDataset(object):
    def __init__(self, args, dataset_dir):
        self._load_annotations(args, dataset_dir)
        #self._load_html_path(dataset_dir)
        #self._mark_ignore_sentece(args)
        self._load_plain_path(dataset_dir)

    @property
    def categories(self):
        return self.__categories

    @property
    def html_paths(self):
        return self.__html_paths

    @property
    def plain_paths(self):
        return self.__plain_paths

    @property
    def annotations(self):
        return self.__annotations

    @annotations.setter
    def annotations(self, annotations):
        self.__annotations = annotations

    def to_c_cls(self, category):
        return self.__category2c_cls.get(category)

    def _load_annotations(self, args, dataset_dir):
        self.__c_clses = sorted(dataset_dir.keys())
        self.__categories = []
        self.__category2c_cls = {}
        self.__annotations = {}

        jobs = []
        for c_cls in self.__c_clses:
            annotation_dir = os.path.join(dataset_dir[c_cls], "annotation")
            for file_path in glob.glob(os.path.join(annotation_dir, "*_dist.json")):
                category = re.match(".*/(.*?)_dist.json", file_path).group(1)

                if args.categories is not None and category not in args.categories:
                    continue

                self.__categories.append(category)
                self.__category2c_cls[category] = c_cls

                jobs.append((category, file_path))

        with Pool(args.parallel) as p, tqdm.tqdm(total=len(jobs), desc="Loading annotations") as t:
            for category, annotation in p.imap_unordered(DataUtils.wrp_load_annotation, jobs):
                self.__annotations[category] = annotation
                t.update()

    def _load_html_path(self, dataset_dir):
        self.__html_paths = defaultdict(dict)
        for category in tqdm.tqdm(self.__categories, desc="Loading html path"):
            c_cls = self.__category2c_cls[category]
            html_dir = os.path.join(dataset_dir[c_cls], "html", category)
            for file_path in glob.glob(os.path.join(html_dir, "*.html")):
                page_id = re.match(".*/(\d*?).html", file_path).group(1)
                self.__html_paths[category][page_id] = file_path

    def _load_plain_path(self, dataset_dir):
        self.__plain_paths = defaultdict(dict)
        for category in tqdm.tqdm(self.__categories, desc="Loading plain path"):
            c_cls = self.__category2c_cls[category]
            plain_dir = os.path.join(dataset_dir[c_cls], "plain", category)
            for file_path in glob.glob(os.path.join(plain_dir, "*.txt")):
                page_id = re.match(".*/(\d*?).txt", file_path).group(1)
                self.__plain_paths[category][page_id] = file_path

    @staticmethod
    def _mark_ignore_sentece_func(inputs):
        # 並列処理用の関数
        parser = IgnoreSentenceParser()

        for page_id, file_path in inputs:
            text = DataUtils.load_file(file_path)
            parser.feed(text)
            parser.reset()

    def _mark_ignore_sentece(self, args):
        for category in tqdm.tqdm(self.__categories, desc="Marking ignore sentence"):
            chunks = [*DataTools.split_array([*self.__html_paths[category].items()], args.parallel)]
            with Pool(args.parallel) as p:
                p.map(self._mark_ignore_sentece_func, chunks)

class TokenizedDataset(object):
    def __init__(self, args, file_dir):
        self._load_annotations(args, file_dir)
        self._load_vocab(file_dir)
        self._load_tokens_path(file_dir)

    @property
    def categories(self):
        return self.__categories

    @property
    def tokens_paths(self):
        return self.__tokens_paths

    @property
    def annotations(self):
        return self.__annotations

    @property
    def vocab(self):
        return self.__vocab

    def _load_annotations(self, args, file_dir):
        self.__categories = []
        self.__annotations = {}

        jobs = []
        for file_path in glob.glob(os.path.join(file_dir, "*/*/*_dist.json")):
            *_, c_cls, category, _ = file_path.split(os.path.sep)

            if args.categories is not None and category not in args.categories:
                continue

            self.__categories.append(category)

            jobs.append((category, file_path))

        with Pool(args.parallel) as p, tqdm.tqdm(total=len(jobs), desc="Loading annotations") as t:
            for category, annotation in p.imap_unordered(DataUtils.wrp_load_annotation, jobs):
                self.__annotations[category] = annotation
                t.update()

    def _load_vocab(self, file_dir):
        self.__vocab = {}

        for category in self.__categories:
            for file_path in glob.glob(os.path.join(file_dir, f"*/{category}/vocab.txt")):
                vocab = DataUtils.load_file(file_path)
                self.__vocab[category] = vocab.splitlines()

    def _load_tokens_path(self, file_dir):
        self.__tokens_paths = defaultdict(dict)

        for category in self.__categories:
            for file_path in glob.glob(os.path.join(file_dir, f"*/{category}/tokens/*.txt")):
                page_id = re.match(".*/(\d*?).txt", file_path).group(1)
                self.__tokens_paths[category][page_id] = file_path
