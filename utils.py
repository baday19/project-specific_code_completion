import json
import pickle

import torch
from transformers import AutoTokenizer, AutoModel
import tiktoken
import numpy as np
import platform

repo_dir = 'repos/'
# repo_dir = 'repocoder_repos/'

class Similarity:
    @staticmethod
    def jaccard_similarity(list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        if intersection == 0 or union == 0:
            return 0
        return float(intersection) / union

    @staticmethod
    def cosine_similarity(embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


class Utils:
    @staticmethod
    def load_pickle(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def dump_pickle(obj, fname):
        with open(fname, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def read_code(fname):
        with open(fname, 'r', encoding='utf8') as f:
            return f.read()

    @staticmethod
    def load_jsonl(fname):
        with open(fname, 'r', encoding='utf8') as f:
            lines = []
            for line in f:
                lines.append(json.loads(line))
            return lines

    @staticmethod
    def dump_jsonl(obj, fname):
        with open(fname, 'w', encoding='utf8') as f:
            for item in obj:
                f.write(json.dumps(item) + '\n')

    @staticmethod
    def iterate_repository(repo):
        base_dir = repo_dir
        pattern = os.path.join(f'{base_dir}/{repo}', "**", "*.py")
        files = glob.glob(pattern, recursive=True)

        skipped_files = []
        loaded_code_files = dict()
        base_dir_list = os.path.normpath(base_dir).split(os.sep)
        for fname in files:
            try:
                code = open(fname, 'r', encoding='utf8').read()
                fpath_tuple = tuple(os.path.normpath(fname).split(os.sep)[len(base_dir_list):])
                loaded_code_files[fpath_tuple]= code
            except Exception as e:
                skipped_files.append((fname, e))
                continue

        if len(skipped_files) > 0:
            print(f"Skipped {len(skipped_files)} out of {len(files)} files due to I/O errors")
            for fname, e in skipped_files:
                print(f"{fname}: {e}")
        return loaded_code_files


class UnixCoder:
    def __init__(self):
        model_name = '/home/username/unixcoder' if platform.system() == 'Linux' else ''
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode_texts(self, texts):
        with torch.no_grad():
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        return embeddings

    def encode_text(self, text):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        return embeddings[0]


class CommonTokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("p50k_base")

    def tokenize(self, text):
        # return self.tokenizer.encode(text)
        return self.tokenizer.encode_ordinary(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

# class DeepSeekCoderTokenizer:

class BlockGroupBuilder:
    def __init__(self, code, k=10):
        self.code = code
        self.sub_blocks = self.split_to_blocks1(code)
        self.k = k
        self.code_lines = code.splitlines()

    def split_to_blocks(self, code):
        lines = code.splitlines()

        res = []
        temp = []
        for line_no, line in enumerate(lines):
            if line != '':
                temp.append((line, line_no))
                continue
            if len(temp) != 0:
                res.append(temp)
            temp = []

        if len(temp) != 0:
            res.append(temp)
        return res

    def split_to_blocks1(self, code):
        codelines = code.splitlines()
        SPLIT_MARK = '@split mark@'
        pre_is_comment = False
        process_lines = []
        for line_no, line in enumerate(codelines):
            curr = line.strip()
            if curr == '':
                if len(process_lines) !=0 and process_lines[-1] != SPLIT_MARK:
                    process_lines.append(SPLIT_MARK)
            elif curr.startswith('#'):
                if pre_is_comment:
                    process_lines.append((line, line_no))
                else:
                    if len(process_lines) !=0 and process_lines[-1] != SPLIT_MARK:
                        process_lines.append(SPLIT_MARK)
                    process_lines.append((line, line_no))
                    pre_is_comment = True
            else:
                process_lines.append((line, line_no))
                pre_is_comment = False

        subchunk_list = []
        temp_subchunk = []
        for i in process_lines:
            if i != SPLIT_MARK:
                temp_subchunk.append(i)
            else:
                if len(temp_subchunk) != 0:
                    subchunk_list.append(temp_subchunk)
                temp_subchunk = []

        if len(temp_subchunk) != 0:
            subchunk_list.append(temp_subchunk)
        
        return subchunk_list

    def get_block_line(self, text):
        return len([i for i in text.splitlines() if i.strip()])

    def build_group_obj(self, group_lines):
        first_line_no = group_lines[0][1]
        last_line_no = group_lines[-1][1]
        return {
            'context': '\n'.join([i[0] for i in group_lines]),
            'first_no': first_line_no,
            'last_no': last_line_no
        }

    def get_block_groups(self):
        block_group_list = []
        for i in range(len(self.sub_blocks)):
            sub_block = self.sub_blocks[i]
            if len(sub_block) < self.k:
                temp_group_arr = sub_block
                j = i + 1
                while len(temp_group_arr) < self.k and j != len(self.sub_blocks):
                    sub_block = self.sub_blocks[j]
                    temp_group_arr.extend(sub_block)
                    j += 1

                block_group_list.append(self.build_group_obj(temp_group_arr[:self.k]))
            else:
                i = 0
                temp_group_arr = []
                while i < len(sub_block):
                    if len(temp_group_arr) == self.k:
                        block_group_list.append(self.build_group_obj(temp_group_arr))
                        temp_group_arr = []
                        i -= int(self.k / 2)
                    temp_group_arr.append(sub_block[i])
                    i += 1
                block_group_list.append(self.build_group_obj(temp_group_arr))
        return block_group_list


