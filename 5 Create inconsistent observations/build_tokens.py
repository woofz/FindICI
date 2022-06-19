from treelib import Node, Tree
from nltk import word_tokenize

from anytree import Node, RenderTree, AsciiStyle, PreOrderIter
from anytree.importer import DictImporter
import pandas as pd


def is_empty(params):
    
    flag = False
    if params:
        flag = True
    return flag


def build_ast(parent, task):

    root = Node(parent)

    for key in list(task.keys()):
        
        node_key = Node(key, parent=root)
        value_element = task[key]
        Node(value_element,parent=node_key)  

    return root

def check_for_nested(node_list):

    for node in node_list:
        if isinstance(node,dict):
            index = node_list.index(node)
            index_parent = index-1
            parent = node_list[index_parent]
            nested = build_ast(parent, node)
            nested_list = [node.name for node in PreOrderIter(nested)]
            nested_list.pop(0)
            node_list[index] = nested_list

    return node_list

def return_tokens(task):

    r = 'AnsibleTask'
    root = build_ast(r,task)
    all_nodes = [node.name for node in PreOrderIter(root)]
    nodes = check_for_nested(all_nodes)

    return nodes

def convert_int_bool_to_str(node_list):

    for node in node_list:
        if (isinstance(node,int)) or (isinstance(node,bool)) or (node == None):
            index = node_list.index(node)
            new_str = str(node)
            node_list[index] = new_str

    return node_list


def rem_nested_lists(node_list):

    for node in node_list:
        if isinstance(node,list):
            index = node_list.index(node)

            joined_string = ",".join(map(str, node))
            node_list[index] = joined_string

    return node_list

def tokenize_task(node_list):

    for node in node_list:
        index = node_list.index(node)
        new_node = word_tokenize(node)
        node_list[index] = new_node

    return node_list

def tokenize_name(task_name):

    seq = [word_tokenize(token) for token in task_name]

    return seq


remove_list = [',', '``','"',"''",'==','{','}','(',"'",'[',']',"'==",')','>','=','-','|','/','+','$','.+',':','!','--','\\','?','@',';','&','#','^','<','*','%','.','=~']

def remove_symbols(sequence):
    
    for obj in sequence:
        for element in obj:
            if element in remove_list:
                obj.remove(element)
            elif "'" in element:
                obj.remove(element)                

    return sequence

def flatten_list(sequence):

    flat_list = [item for sublist in sequence for item in sublist]

    return flat_list

def remove_symbols_simple(sequence):

    for item in sequence:
        if item in remove_list:
            sequence.remove(item)

    return sequence

def rebuild_tokens(sequence):
    from copy import deepcopy
    df = deepcopy(sequence)
    mapped20 = pd.DataFrame()
    mapped20['initial_ast'] = [return_tokens(x) for keys, x in df.items()] 
    mapped20['second_ast'] = [convert_int_bool_to_str(x) for x in mapped20['initial_ast']]
    mapped20['third_ast'] = [rem_nested_lists(x) for x in mapped20['initial_ast']]
    
    # Tokenization
    mapped20['first_tokens'] = [tokenize_task(x) for x in mapped20['third_ast']]
    mapped20['second_tokens'] = [remove_symbols(x) for x in mapped20['first_tokens']]
    mapped20['third_tokens'] = [flatten_list(x) for x in mapped20['second_tokens']]
    
    m20 = mapped20.drop(columns=['initial_ast', 'second_ast', 'first_tokens', 'second_tokens'])
    return m20