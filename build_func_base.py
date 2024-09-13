import glob
import re
from tqdm import tqdm
from tree_sitter_languages import get_language, get_parser
import os

from utils import Utils, UnixCoder, repo_dir


def process_class(node):
    class_block = ''
    name = ''
    for child in node.children:
        if child.type == 'class_body':
            class_block = child.text.decode()
        elif child.type == 'identifier':
            name = child.text.decode()
    def_info = ''
    if class_block in node.text.decode():
        def_info = node.text.decode().replace(class_block, '')
        def_info_lines = [i.strip() for i in def_info.splitlines()]
        def_info = '\n'.join(def_info_lines)
    return {'name': name, 'def_info': def_info}

def process_formal_parameters(params_node):
    id_list = []
    for child in params_node.children:
        if child.type == 'formal_parameter':
            for param_child in child.children:
                if param_child.type == 'identifier':
                    id_list.append(param_child.text.decode())
    
    params_str = ''

    if len(id_list) >= 1:
        params_str = id_list[0]

        for identifier in id_list[1:]:
            params_str += f', {identifier}'
    
    return f'({params_str})'


def process_func(node):
    modifier_type = 'package_private'
    is_static = False
    is_constructor = node.type == 'constructor_declaration'
    name = ''
    params_w_type = ''
    params_wo_type = ''
    func_body = node.text.decode()
    def_info = ''
    block = ''
    for child in node.children:
        if child.type == 'block' or child.type == 'constructor_body':
            block = child.text.decode()
        elif child.type == 'modifiers':
            func_modifiers = child.text.decode()
            if 'static' in func_modifiers:
                is_static = True
            if 'public' in func_modifiers:
                modifier_type = 'public'
            elif 'private' in func_modifiers:
                modifier_type = 'private'
            elif 'protected' in func_modifiers:
                modifier_type = 'protected'
        elif child.type == 'identifier':
            name = child.text.decode()
        elif child.type == 'formal_parameters':
            params_w_type = child.text.decode()
            params_wo_type = process_formal_parameters(child)
    if block in func_body:
        def_info = func_body.replace(block, '').strip()
        def_info_lines = [i.strip() for i in def_info.splitlines()]
        def_info = '\n'.join(def_info_lines)
    
    return {
        'name': name,
        'modifier_type': modifier_type,
        'is_static': is_static,
        'params_w_type': params_w_type,
        'params_wo_type': params_wo_type,
        'func_body': func_body,
        'def_info': def_info,
        'is_constructor': is_constructor
    }


def camel_to_snake(name):
    name = name[0].lower() + name[1:]
    return name

class FuncBaseBuilder:
    def __init__(self, repos):
        self.repos = repos
        self.language = get_language('java')
        self.parser = get_parser('java')
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
                example['class'] = process_class(class_def)

                if not example['func']['is_constructor']:
                    ue = (camel_to_snake(example['class']['name']) if not example['func']['is_static'] else example['class']['name'])  + "." + example['func']['name'] + example['func']['params_wo_type']
                else:
                    ue = example['class']['name'] + ' ' + camel_to_snake(example['class']['name']) + ' = new ' + example['func']['name'] + example['func']['params_wo_type']

                fc = example['class']['def_info'] + ' {\n' + '\n'.join(['    ' + i for i in example['func']['def_info'].splitlines()]) + ';\n}'

                embedding = self.encoder.encode_text(ue)
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
                    'UE_embedding': embedding,
                    'UE': ue,
                    'FC': fc,
                })
            if benchmark is None:
                Utils.dump_pickle(func_database, f'./cache/func_base/{repo}.pkl')
            else:
                Utils.dump_pickle(func_database, f'./cache/func_base/{benchmark}_{repo}.pkl')

    def get_func_list(self, repo_name):
        files_list = glob.glob(os.path.join(self.repo_dir, repo_name, '**/*.java'), recursive=True)
        func_list = []
        for file in files_list:
            file_func_list = self.parse_jfile(file)
            func_list.extend(file_func_list)
        return func_list

    def parse_jfile(self, j_file):
        root = self.parser.parse(open(j_file, 'r', encoding='utf-8').read().encode()).root_node
        package = None
        func_list = []

        def parse_class(class_node):
            class_body = None
            temp_func_list = []
            for child in class_node.children:
                child_type = child.type
                if child_type == 'class_body':
                    class_body = child

            for child in class_body.children:
                child_type = child.type
                if child_type in ['method_declaration', 'constructor_declaration']:
                    temp_func_list.append({'class_def': class_node, 'func_def': child, 'file_path': j_file, 'package': package})
            return temp_func_list


        for child in root.children:
            if child.type == 'package_declaration':
                package = child.children[1].text.decode()
            break
        for child in root.children:
            if child.type == 'class_declaration':
                func_list.extend(parse_class(child))

        # def traverse(node, class_def):
        #     if len(node.children) == 0:
        #         return
        #     for i in node.children:
        #         if i.type == 'package_declaration':
        #             nonlocal package
        #             package = i.children[1].text.decode()
        #         elif i.type == 'class_declaration':
        #             traverse(i, i)
        #         elif i.type in ['constructor_declaration', 'method_declaration']:
        #             if class_def:
        #                 func_list.append({'class_def': class_def, 'func_def': i, 'file_path': j_file, 'package': package})
        #             traverse(i, class_def)
        #         else:
        #             traverse(i, class_def)

        # traverse(root, None)
        return func_list

