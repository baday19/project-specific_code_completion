import os
from summrize.summrize_unit import use_llm
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from utils import Utils, UnixCoder


class SummaryModel:
    def __init__(self, cuda=3):
        self.device = f'cuda:{cuda}'

        model_name = "/home/username/llama3-8b-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(self.device)
        # self.model.generation_config = GenerationConfig.from_pretrained(model_name)
        # self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id

    def summarize_code(self, code):
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": code},
        ]
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        input_tensor = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        outputs = self.model.generate(input_tensor.to(self.model.device), eos_token_id=terminators, max_new_tokens=100,
                                      temperature=0.1,
                                      pad_token_id=self.tokenizer.eos_token_id)
        result = self.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        return result
        # print(summary)


def postprocess_llama(summary):
    summary = summary.lower().replace('here is the summary:', '').replace('summary:', '').replace(
        'summarize the code:', '').strip()
    return summary

def summary_codes(repos, benchmark=None, summary_cuda=3):
    model = SummaryModel(cuda=summary_cuda)
    prompt_template = open('./prompt_java').read()

    for repo in repos:
        func_base = Utils.load_pickle(f'./cache/func_base/{repo if benchmark is None else benchmark + "_" + repo}.pkl')
        summary_list = []
        for func in tqdm(func_base):
            body = func['metadata']['func_body']
            prompt = prompt_template.replace('@{}@', body)
            func_summary = model.summarize_code(prompt).strip()
            func_summary = postprocess_llama(func_summary)
            summary_list.append(func_summary)
            # print(func_summary)

        Utils.dump_pickle(summary_list, f'./cache/func_base/{repo if benchmark is None else benchmark + "_" + repo}_summary.pkl')


def summary_code_use_llm(repos):
    prompt_template = open('./prompt_java').read()

    for repo in repos:
        func_base = Utils.load_pickle(f'./cache/func_base/{repo}.pkl')
        summary_list = []
        for func in tqdm(func_base):
            body = func['metadata']['func_body']
            func_summary = use_llm(prompt_template.replace('@{}@', body))
            summary_list.append(func_summary)
            # print(func_summary)

        Utils.dump_pickle(summary_list, f'./cache/func_base/{repo}_summary.pkl')


def summary_one_code_use_llm(code):
    prompt_template = open('./prompt_java').read()
    func_summary = use_llm(prompt_template.replace('@{}@', code))
    return func_summary


def process_not_has_e3(example):
    lines = [i.strip() for i in example.splitlines() if i.strip()]
    res = []
    # A:
    for line in lines:
        if not line.startswith('A:'):
            res.append(line)
        else:
            break
    if len(res) == 0:
        return None
    else:
        return '\n'.join(res).strip()


def encode_summaries(repos, benchmark=None):
    unixcoder_enc = UnixCoder()
    for repo in repos:
        summary_list = Utils.load_pickle(f'./cache/func_base/{repo if benchmark is None else benchmark + "_" + repo}_summary.pkl')
        func_base = Utils.load_pickle(f'./cache/func_base/{repo if benchmark is None else benchmark + "_" + repo}.pkl')
        for idx in tqdm(range(len(summary_list))):
            func_item = func_base[idx]
            summary = summary_list[idx]
            # if 'Example-3' not in summary:
            #     temp = process_not_has_e3(summary)
            # else:
            #     temp = summary.split('Example-3')[0].strip()
            # temp = temp if temp is not None else func_item['doc']
            temp = summary
            func_item['summary'] = temp
            func_item['summary_vec'] = unixcoder_enc.encode_text(temp)
        Utils.dump_pickle(func_base, f'./cache/func_base/{repo if benchmark is None else benchmark + "_" + repo}_with_summary.pkl')


