from utils import CommonTokenizer, Similarity, Utils, repo_dir
import numpy as np
import os

class SnippetPromptBuilder:

    def __init__(self, examples, repos, window_size, benchmark_path):
        self.window_size = window_size
        self.examples = examples
        self.repos_source_code = {}
        self.tokenizer = CommonTokenizer()
        self.repos_databses = {}
        self.seperator = '# ' + '-' * 50
        for repo in repos:
            self.repos_source_code[repo] = Utils.iterate_repository(repo)
            self.repos_databses[repo] = Utils.load_pickle(f'./cache/snippet_base/{repo}.pkl')
        task_path = benchmark_path
        tasks = Utils.load_jsonl(task_path)
        self.tasks_by_task_id = {task['metadata']['task_id']: task for task in tasks}

    def get_query(self, task):
        repo_name =  task['metadata']['task_id'].split('/')[0]
        source_code = self.repos_source_code[repo_name]
        fpath_tuple = tuple(task['metadata']['fpath_tuple'])
        line_no = task['metadata']['line_no']
        original_code = source_code[fpath_tuple]
        code_lines = original_code.splitlines()
        context_start_lineno = task['metadata']['context_start_lineno']
        start_line_no = max(context_start_lineno, line_no - self.window_size)
        window_lines = [i for i in code_lines[start_line_no:line_no]]
        context = '\n'.join(window_lines)
        tokinzerd = self.tokenizer.tokenize(context)
        return tokinzerd
    
    def get_query_by_cd(self, task):
        delta_size = self.window_size // 2
        repo_name =  task['metadata']['task_id'].split('/')[0]
        source_code = self.repos_source_code[repo_name]
        fpath_tuple = tuple(task['metadata']['fpath_tuple'])
        line_no = task['metadata']['line_no']
        original_code = source_code[fpath_tuple]
        code_lines = original_code.splitlines()
        context_start_lineno = task['metadata']['context_start_lineno']
        start_line_no = max(context_start_lineno, line_no - self.window_size)
        for sample in [task['choices'][i]['text'] for i in range(len(task['choices']))]:
            sample_lines = [i for i in sample.splitlines() if i.strip()]
            new_code_lines = code_lines[:line_no] + sample_lines
            end_line_no = min(len(new_code_lines), line_no + self.window_size - delta_size)
            window_lines = [i for i in new_code_lines[start_line_no:end_line_no] if i.strip()]
            if not window_lines:
                continue
        window_lines = [i for i in code_lines[start_line_no:line_no]]
        context = '\n'.join(window_lines)
        tokinzerd = self.tokenizer.tokenize(context)
        return tokinzerd

    def run(self, use_code_draft=False):
        res = []
        for task in self.examples:
            query = self.get_query(task) if not use_code_draft else self.get_query_by_cd(task)
            # ensure enough context length for api information
            ret_length = 2000 if not use_code_draft else 1600
            repo_name =  task['metadata']['task_id'].split('/')[0]
            topk_context = self.search_top_k(task, query, repo_name)
            res_example = self.build_prompt(use_code_draft, task, topk_context, ret_length)
            res.append(res_example)
        return res


    def build_prompt(self, use_cd, query_line, top_k_context, ret_len):
        task_id = query_line['metadata']['task_id']
        task = self.tasks_by_task_id[task_id]
        old_prompt = task['prompt']
        top_k_context = top_k_context
        new_prompt, chosen_context = self._build_prompt(use_cd, old_prompt, top_k_context, ret_len)
        new_prompt_line = {
            'prompt': new_prompt,
            'ret_prompt': new_prompt.replace(old_prompt, ''),
            'metadata': task['metadata'],
        }
        return new_prompt_line


    def _is_context_after_hole(self, repo_embedding_line, query_line):
        hole_fpath_tuple = tuple(query_line['metadata']['fpath_tuple'])
        context_is_not_after_hole = []
        for metadata in repo_embedding_line['metadata']:
            if tuple(metadata['fpath_tuple']) != hole_fpath_tuple:
                context_is_not_after_hole.append(True)
                continue
            # now we know that the repo line is in the same file as the hole
            if metadata['end_line_no'] <= query_line['metadata']['context_start_lineno']:
                context_is_not_after_hole.append(True)
                continue
            context_is_not_after_hole.append(False)
        return not any(context_is_not_after_hole)
        
    def search_top_k(self, query_line, query_embedding, repo):
        top_k_context = []
        query_embedding = query_embedding
        repo_embedding_lines = self.repos_databses[repo]
        for repo_embedding_line in repo_embedding_lines:
            if self._is_context_after_hole(repo_embedding_line, query_line):
                continue
            repo_line_embedding = np.array(repo_embedding_line['data'][0]['embedding'])
            similarity_score = Similarity.jaccard_similarity(query_embedding, repo_line_embedding)
            top_k_context.append((repo_embedding_line, similarity_score))
        top_k_context = sorted(top_k_context, key=lambda x: x[1], reverse=False)[-10:]
        return top_k_context
            
            
    def _make_a_block(self, retrieved_context):
        content, sim_score = retrieved_context
        metadata = content['metadata']
        # put the file path in the comment
        assert metadata[0]['fpath_tuple'][0] == metadata[0]['repo']
        f_paths = ['/'.join(x['fpath_tuple'][1:]) for x in metadata]
        f_paths_str = '\n'.join([f'# {f_path}' for f_path in f_paths])
        f_path_comment = f'# the below code fragment can be found in:'
        # put code lines in the comment
        content_lines = content['context'].splitlines()
        content_lines_comment = [f'# {line}' for line in content_lines]
        # aggregate the comment and the code lines
        
        block_str = '\n'.join([f_path_comment, f_paths_str, self.seperator] + content_lines_comment + [self.seperator]) + '\n'
        tokenized_block = self.tokenizer.tokenize(block_str)
        token_len = len(tokenized_block)
        return block_str, token_len

    def _make_an_extended_block(self, retrieved_context):
        content, sim_score = retrieved_context
        metadata = content['metadata']
        # put the file path in the comment
        assert metadata[0]['fpath_tuple'][0] == metadata[0]['repo']
        f_paths = ['/'.join(x['fpath_tuple'][1:]) for x in metadata]
        f_paths_str = '\n'.join([f'# {f_path}' for f_path in f_paths])
        f_path_comment = f'# the below code fragment can be found in:'
        # put code lines in the comment
        original_code = open(os.path.join(repo_dir, *metadata[0]['fpath_tuple']), 'r').read()
        code_lines = original_code.splitlines()
        end_line_no = metadata[0]['end_line_no']
        window_size = metadata[0]['window_size']
        slice_size = metadata[0]['slice_size']
        new_end_line_no = min(end_line_no + window_size // slice_size, len(code_lines))
        new_start_line_no = max(0, new_end_line_no - window_size)
        content_lines = code_lines[new_start_line_no:new_end_line_no]
        content_lines_comment = [f'# {line}' for line in content_lines]
        # aggregate the comment and the code lines
        block_str = '\n'.join([f_path_comment, f_paths_str, self.seperator] + content_lines_comment + [self.seperator]) + '\n'
        tokenized_block = self.tokenizer.tokenize(block_str)
        token_len = len(tokenized_block)
        return block_str, token_len

    def _build_prompt(self, use_cd, prompt, top_k_context, ret_len):
        prepend_context = "# Here are some relevant code fragments from other files of the repo:\n"
        prepend_context += self.seperator + '\n'
        current_token_length = 20  # the length of the head_prompt, same for codex and codegen tokenizer
        prepend_blocks = []
        chosen_context = []
        make_block_func = self._make_an_extended_block if not use_cd else self._make_a_block
        for retrieved_context in top_k_context[::-1]:
            if len(chosen_context) >= 10:
                break
            block_str, token_len = make_block_func(retrieved_context)
            if current_token_length + token_len < ret_len:
                prepend_blocks.insert(0, block_str)
                current_token_length += token_len
                chosen_context.append(retrieved_context)
            else:
                continue
        prepend_context += ''.join(prepend_blocks)  # all the blocks already have a line break at the end
        return prepend_context + '\n' + prompt, chosen_context
