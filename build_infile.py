import glob
import os.path

from tqdm import tqdm
from tree_sitter_languages import get_language, get_parser

import utils
from utils import Utils, CommonTokenizer


class DefinitionNode:
    def __init__(self, node=None):
        self.children = []
        self.data = node
        self.doc = self._process_node2doc(node) if node is not None else None

    def _process_node2doc(self, node):
        if node.type == 'decorated_definition':
            node = node.children[1]
        if node.type == 'function_definition':
            return process_func(node)['sign']
        elif node.type == 'class_definition':
            return process_class(node)['sign'] + ':'

    def add_child(self, node):
        self.children.append(node)


def process_class(node):
    name = ''
    arg_list = ''
    for child in node.children:
        if child.type == 'argument_list':
            arg_list = child.text.decode()
        elif child.type == 'identifier':
            name = child.text.decode()
    class_tag = f'class {name}{arg_list}'
    return {'name': name, 'arg_list': arg_list, 'sign': class_tag}


def process_func(node):
    name = ''
    params = ''
    return_type = ''
    for child in node.children:
        if child.type == 'parameters':
            params = child.text.decode()
        elif child.type == 'identifier':
            name = child.text.decode()
        elif child.type == 'type':
            return_type = child.text.decode()
    sign = f'def {name}{params}{" -> " + return_type if return_type != "" else ""}'
    return {'name': name, 'params': params, 'return_type': return_type, 'sign': sign}


class InFileContextProcessor:
    def __init__(self):
        self.tokenizer = CommonTokenizer()
        self.language = get_language('python')
        self.parser = get_parser('python')
        self.repo_dir = utils.repo_dir

    def build_structured_context(self, code, token_limit=512):
        def_root = DefinitionNode()
        root = self.parser.parse(code.encode()).root_node
        res_list = []

        def traverse(node, dad):
            if node.child_count == 0:
                return
            else:
                if node.type in ['class_definition', 'function_definition']:
                    temp = DefinitionNode(node)
                    dad.add_child(temp)
                    for child in node.children:
                        traverse(child, temp)
                else:
                    for child in node.children:
                        traverse(child, dad)

        traverse(root, def_root)

        def traverse_deftree(node, depth=0):
            if node.doc is not None:
                res_list.append('\n'.join(['    ' * (depth - 1) + line for line in node.doc.splitlines()]))
            depth += 1
            for child in node.children:
                traverse_deftree(child, depth)

        traverse_deftree(def_root)
        structured_prompt = ''
        prompt_len = 0
        structured_prompt_list = []
        for item in res_list[::-1]:
            curr_item_len = len(self.tokenizer.tokenize(item))
            prompt_len += curr_item_len
            if prompt_len > token_limit:
                break
            structured_prompt_list.append(item)
        for item in structured_prompt_list[::-1]:
            structured_prompt += '\n'.join(['# ' + line for line in item.splitlines()]) + '\n'

        return structured_prompt

    def get_infile_context(self, example, simple_token_limit=1048, use_structed=False, structed_prompt_len=512):
        o_prompt = example['prompt']
        fpath = os.path.join(self.repo_dir, *example['metadata']['fpath_tuple'])
        row = example['metadata']['line_no']
        col = example['metadata']['col']
        file_content = open(fpath, 'r', encoding='utf8').read()
        code_lines = file_content.splitlines()
        pre_code_lines = code_lines[:row] + [code_lines[row][:col]]

        simple_in_len = 0
        curr_row = len(pre_code_lines) - 1
        while curr_row >= 0:
            curr_line = pre_code_lines[curr_row]
            curr_line_len = len(self.tokenizer.tokenize(curr_line))
            simple_in_len += curr_line_len
            if simple_in_len > simple_token_limit:
                break
            curr_row -= 1

        simple_start_lineno = curr_row + 1
        simple_prompt = '\n'.join(pre_code_lines[simple_start_lineno:])
        if simple_start_lineno > 0 and use_structed:
            structured_context = self.build_structured_context('\n'.join(pre_code_lines[:simple_start_lineno]),
                                                               token_limit=structed_prompt_len)
            return structured_context + '\n' + simple_prompt
        return simple_prompt


def build_infile(in_file, out_file, infile_len=4096):
    examples = Utils.load_jsonl(in_file)
    infile_context_processor = InFileContextProcessor()
    samples = []
    for example in tqdm(examples):
        simple_prompt = infile_context_processor.get_infile_context(example, simple_token_limit=infile_len,
                                                                    use_structed=False)
        example['infile_prompt'] = simple_prompt
        samples.append(example)
    Utils.dump_jsonl(samples, out_file)
