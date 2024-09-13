import glob
import re
from tqdm import tqdm
from tree_sitter_languages import get_language, get_parser
import os

from utils import Utils, UnixCoder, repo_dir


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

def process_params(params_node):
    id_list = []
    for child in params_node.children:
        if child.type == 'identifier':
            if child.text.decode().strip() != 'self':
                id_list.append(child.text.decode())
        else:
            for param_child in child.children:
                if param_child.type == 'identifier':
                    id_list.append(param_child.text.decode())
        # if child.type not in ['(', 'typed_parameter', ')', 'identifier', ',',
        #                       'default_parameter', 'typed_default_parameter', 'list_splat_pattern', 'dictionary_splat_pattern', 'comment']:
        #     print(child.type)
    params_str = ''

    if len(id_list) >= 1:
        params_str = id_list[0]

        for identifier in id_list[1:]:
            params_str += f', {identifier}'
    
    return f'({params_str})'


def process_func(node):
    name = ''
    params = ''
    params_wo_type = ''
    return_type = ''
    for child in node.children:
        if child.type == 'parameters':
            params = child.text.decode()
            params_wo_type = process_params(child)
        elif child.type == 'identifier':
            name = child.text.decode()
        elif child.type == 'type':
            return_type = child.text.decode()
    sign = f'def {name}{params}{" -> " + return_type if return_type != "" else ""}'
    return {'name': name, 'params': params, 'params_wo_type': params_wo_type, 'return_type': return_type, 'sign': sign}

def camel_to_snake(camel_str):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', camel_str).lower()

class FuncBaseBuilder:
    def __init__(self, repos):
        self.repos = repos
        self.language = get_language('python')
        self.parser = get_parser('python')
        self.encoder = UnixCoder()
        self.repo_dir = repo_dir

    def build(self, benchmark=None):
        for repo in self.repos:
            func_list = self.get_func_list(repo_name=repo)
            func_database = []
            for example in tqdm(func_list):
                func_def = example['func_def']
                class_def = example['class_def']
                example['func'] = process_func(func_def)
                if class_def:
                    example['class'] = process_class(class_def)
                    if example['func']['name'] == '__init__':
                        doc = example['class']['name'] + example['func']['params_wo_type']
                    else:
                        doc = camel_to_snake(example['class']['name']) + "." + example['func']['name'] + example['func']['params_wo_type']
                    info = example['class']['sign'] + ':\n    ' + example['func']['sign']
                else:
                    example['class'] = None
                    doc = example['func']['name'] + example['func']['params_wo_type']
                    info = example['func']['sign']
                
                embedding = self.encoder.encode_text(doc)
                fpath = tuple([i for i in example['file_path'].replace(self.repo_dir, '').split('/') if i.strip()])
                func_body = func_def.text.decode()
                metadata = {
                    'func': example['func'],
                    'func_body': func_body,
                    'class': example['class'],
                    'lineno': func_def.start_point[0]
                }
                func_database.append({
                    'fpath': fpath,
                    'metadata': metadata,
                    'embedding': embedding,
                    'doc': doc,
                    'info': info,
                })
            if benchmark is None:
                Utils.dump_pickle(func_database, f'./cache/func_base/{repo}.pkl')
            else:
                Utils.dump_pickle(func_database, f'./cache/func_base/{benchmark}_{repo}.pkl')

    def get_func_list(self, repo_name):
        files_list = glob.glob(os.path.join(self.repo_dir, repo_name, '**/*.py'), recursive=True)
        func_list = []
        for file in files_list:
            file_func_list = self.parse_pyfile(file)
            func_list.extend(file_func_list)

        return func_list

    def parse_pyfile(self, py_file):
        root = self.parser.parse(open(py_file, 'r', encoding='utf-8').read().encode()).root_node
        func_list = []

        def traverse(node, class_def):
            if len(node.children) == 0:
                return
            for i in node.children:
                if i.type == 'class_definition':
                    traverse(i, i)
                elif i.type == 'function_definition':
                    func_list.append({'class_def': class_def, 'func_def': i, 'file_path': py_file})
                    traverse(i, class_def)
                else:
                    traverse(i, class_def)

        traverse(root, None)
        return func_list


