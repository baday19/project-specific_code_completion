import os

from build_func_prompt import FuncPromptBuilder
from build_snippet_prompt import SnippetPromptBuilder
from summarize_code import summary_codes, encode_summaries
from utils import repo_dir, Utils, CommonTokenizer
from build_infile import build_infile
from build_func_base import FuncBaseBuilder
from build_snippet_base import SnippetBaseBuilder


def process_infile(in_file, out_file, context_len=1000):
    build_infile(in_file, out_file, infile_len=context_len)

def build_snippet_database(repos):
    snippet_base_builder = SnippetBaseBuilder(repos, 20, 2, CommonTokenizer())
    snippet_base_builder.build()

def build_function_database(repos, benchmark=None, summary_cuda=3):
    func_base_builder = FuncBaseBuilder(repos)
    func_base_builder.build(benchmark=benchmark)
    summary_codes(repos, benchmark=benchmark, summary_cuda=summary_cuda)
    encode_summaries(repos, benchmark=benchmark)

def build_snippet_prompt(repos, examples, use_code_draft):
    snippet_prompt_builder = SnippetPromptBuilder(examples, repos, 20, 'datasets/projbench_java.jsonl')
    res_file = snippet_prompt_builder.run(use_code_draft)
    return res_file

def build_func_prompt(repos, examples, cache_file='./cache/func_retrieval/retrieval.pkl', k=4, summary_cuda=3):
    func_builder = FuncPromptBuilder(repos, summary_cuda=summary_cuda)
    new_examples = func_builder.run(examples, use_doc=True, use_summary=True, k=k)
    Utils.dump_pickle(new_examples, cache_file)
    for example in new_examples:
        temp_example = example
        del temp_example['func_context']
    return new_examples

def build_code_draft_prompt(repos, examples):
    build_snippet_database(repos)
    res_file = build_snippet_prompt(repos, examples, use_code_draft=False)
    Utils.dump_jsonl(res_file, './prompts/code_draft.jsonl')

def build_target_prompt(repos, examples, bench_path, mode='full'):
    similar_prompt = build_snippet_prompt(repos, examples, use_code_draft=True)
    similar_tasks_by_task_id = {task['metadata']['task_id']: task for task in similar_prompt}
    api_prompt = build_func_prompt(repos, examples)
    api_tasks_by_task_id = {task['metadata']['task_id']: task for task in api_prompt}
    tasks = Utils.load_jsonl(bench_path)
    for task in tasks:
        task_id = task['metadata']['task_id']
        o_prompt = task['prompt']
        similar_info = similar_tasks_by_task_id[task_id]['ret_prompt']
        api_info = api_tasks_by_task_id[task_id]['func_prompt']
        if mode == '-uer':
            api_info = api_tasks_by_task_id[task_id]['func_detail']['UEs']
        elif mode == '-fsr':
            api_info = api_tasks_by_task_id[task_id]['func_detail']['docstrings']
        new_prompt = f"""{api_info}\n\n{similar_info}\n\n{o_prompt}"""
        task['prompt'] = new_prompt
    return tasks
    
def build_target_code_prompt(repos, examples_path, mode='full'):
    code_draft_path = f'predictions/code_draft.jsonl'
    code_draft = Utils.load_jsonl(code_draft_path)
    target_code_prompts = build_target_prompt(repos, code_draft, examples_path, mode=mode)
    Utils.dump_jsonl(target_code_prompts, f'prompts/target-{mode}.jsonl')


def mk_dir(path): # path是指定文件夹路径
    if os.path.isdir(path):
        # print('文件夹已存在')
        pass
    else:
        os.makedirs(path)

if __name__ == '__main__':
    summary_cuda = 5 # 用于摘要的设备
    entries = os.listdir(repo_dir)
    repos = [entry for entry in entries if os.path.isdir(os.path.join(repo_dir, entry))]
    print(repos)
    examples_path = 'datasets/projbench_java.jsonl'

    # build api database
    build_function_database(repos, summary_cuda=summary_cuda)

    # build code_draft: prompts/code_draft.jsonl
    build_code_draft_prompt(repos, Utils.load_jsonl(examples_path))

    # search similar info and api info, then generate prompt: prompts/target-mode.jsonl
    build_target_code_prompt(repos, examples_path, mode='full')



