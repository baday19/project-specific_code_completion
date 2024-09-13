from collections import defaultdict
from utils import Utils, CommonTokenizer


class SnippetBaseBuilder:
    def __init__(self, repos, window_size, slice_size, tokenizer):
        self.repos = repos
        self.window_size = window_size
        self.slice_size = slice_size
        self.slice_step = 1 if window_size // slice_size == 0 else window_size // slice_size
        self.tokenizer = tokenizer

    def build(self):
        for repo in self.repos:
            source_code_files = Utils.iterate_repository(repo)
            mergerd_code_windows = self.build_windows(source_code_files, repo)

            # vectorize
            for i in mergerd_code_windows:
                tokenized = self.tokenizer.tokenize(i['context'])
                i['context'] = tokenized
            Utils.dump_jsonl(mergerd_code_windows, f'./cache/snippet_base/{repo}.jsonl')


    def build_windows(self, source_code_files, repo):
        all_code_windows = []
        for fpath_tuple, code in source_code_files.items():
            all_code_windows += self._buid_windows_for_a_file(fpath_tuple, code, repo)
        merged_code_windows = self._merge_windows_with_same_context(all_code_windows)
        return merged_code_windows
    
    def _buid_windows_for_a_file(self, fpath_tuple, code, repo):
        code_windows = []
        code_lines = code.splitlines()
        delta_size = self.window_size // 2
        for line_no in range(0, len(code_lines), self.slice_step): # line_no starts from 0
            start_line_no = max(0, line_no - delta_size)
            end_line_no = min(len(code_lines), line_no + self.window_size - delta_size)
            window_lines = [i for i in code_lines[start_line_no:end_line_no]]
            if not window_lines:  # all empty lines
                continue
            window_text = '\n'.join(window_lines)
            code_windows.append({
                'context': window_text,
                'metadata': {
                    'fpath_tuple': fpath_tuple,
                    'line_no': line_no,
                    'start_line_no': start_line_no,
                    'end_line_no': end_line_no,
                    'window_size': self.window_size,
                    'repo': repo,
                    'slice_size': self.slice_size,
                }
            })
        return code_windows
    
    def _merge_windows_with_same_context(self, code_windows):
        merged_code_windows = defaultdict(list)
        for code_window in code_windows:
            context = code_window['context']
            metadata = code_window['metadata']
            merged_code_windows[context].append(metadata)
        json_lines = []
        for context, metadata_list in merged_code_windows.items():
            json_lines.append({
                'context': context,
                'metadata': metadata_list
            })
        return json_lines


