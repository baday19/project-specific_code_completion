import heapq

import numpy as np
from tqdm import tqdm

from utils import UnixCoder, Utils, CommonTokenizer, Similarity
from summarize_code import SummaryModel, postprocess_llama


class FuncPromptBuilder:
    def __init__(self, repos, summary_cuda=3):
        self.encoder = UnixCoder()
        self.tokenizer = CommonTokenizer()
        self.repos = repos
        self.func_database_dict = dict()
        self.summary_model = SummaryModel(summary_cuda)
        for repo in repos:
            self.func_database_dict[repo] = Utils.load_pickle(f'./cache/func_base/{repo}_with_summary.pkl')

    def get_last_line(self, example):
        task_type = example['metadata']['task_type']
        prompt = example['prompt'].splitlines()
        # model_out = example['model_output'].splitlines()
        pred_res = [line for line in example['pred_res'].splitlines() if line.strip()]
        # print(prompt)
        if task_type == 'sl':
            if len(pred_res) == 0:
                return prompt[-1]
            last_line = prompt[-1] + pred_res[0]
        else:
            if len(pred_res) == 0:
                return ''
            last_line = pred_res[0]
        return last_line

    def get_last_lines(self, example):
        task_type = example['metadata']['task_type']
        prompt = example['prompt'].splitlines()
        # model_out = example['model_output'].splitlines()
        pred_res = [line for line in example['pred_res'].splitlines() if line.strip()]
        if task_type == 'sl':
            if len(pred_res) == 0:
                return prompt[-1]
            last_line = prompt[-1] + pred_res[0]
            last_lines = '\n'.join([last_line] + pred_res[1:])
        else:
            if len(pred_res) == 0:
                return ''
            last_lines = '\n'.join(pred_res)
        return last_lines

    def get_topk_func(self, in_doc, database, example, k=3):
        in_embedding = self.encoder.encode_text(in_doc)
        scores = []
        input_fpath = tuple(example['metadata']['fpath_tuple'])
        input_lineno = example['metadata']['context_start_lineno']
        for i in database:
            data_fpath = tuple(i['fpath'])
            data_lineno = i['metadata']['lineno']
            if data_fpath == input_fpath and data_lineno > input_lineno:
                continue
            embedding = i['UE_embedding']
            sim = Similarity.cosine_similarity(in_embedding, embedding)
            scores.append(sim)
        idxs = heapq.nlargest(k, range(len(scores)), scores.__getitem__)
        # idx = scores.index(max(scores))
        res = [database[i] for i in idxs]
        return res

    def get_topk_func_by_summary(self, in_doc, database, k=3):
        # summary = summary_one_code_use_llm(in_doc)
        prompt_template = open('./prompt_java').read()
        summary = self.summary_model.summarize_code(prompt_template.replace('@{}@', in_doc)).strip()
        summary = postprocess_llama(summary)
        in_embedding = self.encoder.encode_text(summary)
        scores = []
        for i in database:
            embedding = i['summary_vec']
            sim = Similarity.cosine_similarity(in_embedding, embedding)
            scores.append(sim)
        idxs = heapq.nlargest(k, range(len(scores)), scores.__getitem__)
        # idx = scores.index(max(scores))
        res = [database[i] for i in idxs]
        return res, summary

    def build_new_prompt(self, func_list):
        new_prompt = ''
        for func in func_list:
            func_path = '/'.join(func['fpath'])
            func_prompt = func_path + '\n' + func['FC']
            func_prompt = '\n'.join(['# ' + i for i in func_prompt.splitlines()])
            func_prompt = func_prompt + '\n\n'
            new_prompt += func_prompt
        return new_prompt

    def process_example(self, example, use_doc=True, use_summary=True, k=4):
        repo = example['metadata']['task_id'].split('/')[0]
        ret_funcs = []
        ret_funcs0 = []
        ret_funcs1 = []
        summary_query = ''
        if self.get_last_line(example).strip():  
            if use_doc:
                ret_funcs0 = self.get_topk_func(self.get_last_line(example).strip(), self.func_database_dict[repo], k=8, example=example)
                ret_funcs.extend(ret_funcs0[:k])
            if use_summary:
                ret_funcs1, summary_query = self.get_topk_func_by_summary(self.get_last_lines(example).strip(), self.func_database_dict[repo], k=8)
                ret_funcs.extend(ret_funcs1[:k])
        new_prompt_prefix = self.build_new_prompt(ret_funcs)
        doc_prompt = self.build_new_prompt(ret_funcs0[:k])
        summary_prompt = self.build_new_prompt(ret_funcs1[:k])
        new_prompt = new_prompt_prefix
        # new_prompt = new_prompt_prefix + example['prompt']
        example['func_context'] = (ret_funcs0, ret_funcs1)
        example['func_prompt'] = new_prompt
        example['func_detail'] = {'UEs': doc_prompt, 'docstrings': summary_prompt}
        example['summary_query'] = summary_query
        return example

    def run(self, examples, use_doc=True, use_summary=True, k=4):
        new_examples = []
        for example in tqdm(examples):
            new_example = self.process_example(example, use_doc=use_doc, use_summary=use_summary, k=k)
            new_examples.append(new_example)
        return new_examples


