import pandas as pd
import numpy as np
from copy import deepcopy
from typing import Union, Dict, Any
import pickle
import random
import typing
from build_tokens import convert_int_bool_to_str, flatten_list, rem_nested_lists, tokenize_task, tokenize_name, is_empty, build_ast, return_tokens, remove_symbols, rebuild_tokens
from nltk import word_tokenize
from data import module_parameters

def pick_random_method(task_df):
    
    task = task_df.sample(n=1)
    task_descr = ' '.join(task.loc[task.index[0],'third_tokens'])
    task_tuple = (task.index[0],task_descr)
    
    return task_tuple

def drop_same(similars,id):
    for elem in similars:
        if elem[0] == id:
            similars.remove(elem)
    
    return similars        

def get_tasks_from_simlist(similar_list,tasks_df):
    
    name_list = []
    for elem in similar_list:
        row_id = elem[0]
        task_name = ' '.join(tasks_df.loc[row_id,'token_task_names'])
        name_list.append(task_name)
        
    return name_list

def find_task_names_by_id(similar_list,tagged_names):
    
    name_list = []
    for elem in similar_list:
        row_id = elem[0]
        task_name = tagged_names[row_id].words
        name_list.append((task_name,row_id))
        
    return name_list    

def find_similar_names(similar_task_names,model_names):
    
    similar_of_similars = []
    for elem,row_id in similar_task_names:
        elem_similars = model_names.docvecs.most_similar(positive=[model_names.infer_vector(elem)],topn=11)
        elem_sims = drop_same(elem_similars,row_id)
        similar_of_similars.append((elem_sims,row_id))
    
    return similar_of_similars    

def check_random_found(rand_id,similar_names):
    
    found = ''
    for tup in similar_names:
        # found = False
        lst = tup[0]
        index = similar_names.index(tup)
        for tup2 in lst:
            found = False
            if tup2[0] == rand_id:
                found = True
                acc = tup2[1]
                print('The randomly selected element was found at the '+str(index)+ ' position with accuracy' + str(round(acc,2)))

remove_list = [',', '``','"',"''",'==','{','}','(',"'",'[',']',"'==",')','>','=','-','|','/','+','$','.+',':','!',
               '--','\\','?','@',';','&','#','^','<','*','%','.','=~','{{','}}',]

def remove_symbols_simple(sequence):

    for item in sequence:
        if item in remove_list:
            sequence.remove(item)

    return sequence

def mutate_descriptions_old(method, tasks_df):
    
    """ This method mutates the methods descriptions column. It replaces the implementation of the ansible module by randomly selecting another """
    
    module_used = method[1]
    filtered_df = tasks_df[tasks_df['mod_keys_found_string'] != module_used]
    random_method = filtered_df.sample(n=1)['third_tokens'].tolist()[0][1:]
    new_method = method[:1]
    new_method +=  random_method    
    
    return new_method    

def mutate_descriptions2(method, tasks_df):
    
    """ This method mutates the methods descriptions column. It replaces the implementation of the ansible module by randomly selecting another """
    
    module_list = list(tasks_df['mod_keys_found_string'].unique())
    module_used = list(set(module_list) & set(method))[0]
    # print(module_used)
    filtered_df = tasks_df[tasks_df['mod_keys_found_string'] == module_used]
    random_method = filtered_df.sample(n=1)['third_tokens'].tolist()[0][1:]
    new_method = method[:1]
    new_method +=  random_method    
    
    return new_method    

def mutate_params(*, normal_module: dict, module_used: str, task_name: str) -> Dict[str, Any]:
    """
    This method mutates a given module description to another, by changing or adding a random parameter and documenting
    it in the Task Name. Specific changes were made for some modules, such as 'command' and
    Args:
        normal_module: Unmutated module description
        module_used: Used module (i.e.: service, gather_facts)
        task_name: Task name (i.e.: restart datadog-agent)
    Returns:
        dict: returns a dictionary with two keys:
                - task_body: contains the mutated task body;
                - task_name: contains the mutated task title.
    """
    module = normal_module
    module_params = typing.cast(Dict[str, Any], (module_parameters[module_used]))

    if type(module) in (str, bool):
        mutable_module_params = None
    else:
        mutable_module_params = set(module.keys()).intersection(set(module_params.keys()))

    print(module_params)
    if not mutable_module_params or module_used == 'command':
        # Adding a random parameter to the data
        if module_used not in module_parameters.keys():
            return {'task_body': module, 'task_name': task_name}
        param_to_add = random.choice(list(module_parameters.get(module_used)))
        param_value = random.choice(list(module_parameters.get(module_used).get(param_to_add)))
        # getting the mutated description for the parameter
        desc_param_value = [x for x in list(module_parameters.get(module_used).get(param_to_add)) if x != param_value]
        m_desc = param_to_add + ' ' + str(desc_param_value[0])

        task_name += ' ' + m_desc
        if type(module) in (str, bool):
            mutated_dict = dict()
            mutated_dict.update({param_to_add: param_value})
            return {'task_body': mutated_dict, 'task_name': task_name}
        if module is None:
            print("module is none.")
        module.update({param_to_add: param_value})
        return {'task_body': module, 'task_name': task_name}

    param_to_mutate = random.choice(list(mutable_module_params))
    mutable_param_values = module_params[param_to_mutate]
    gen = [i for i in mutable_param_values if i != module[param_to_mutate]]
    mutated_value = random.choice(gen)

    mutated_module = module
    old_param = mutated_module[param_to_mutate]
    mutated_module[param_to_mutate] = mutated_value

    task_name += ' ' + param_to_mutate + ' ' + str(old_param)
    return {'task_body': mutated_module, 'task_name': task_name}

def rebuild_tokens(descriptions, task_names: Union[list, None] = None):
    from copy import deepcopy
    df = deepcopy(descriptions)
    mapped20 = pd.DataFrame()
    mapped20['initial_ast'] = [return_tokens(x) for keys, x in df.items()] 
    mapped20['second_ast'] = [convert_int_bool_to_str(x) for x in mapped20['initial_ast']]
    mapped20['third_ast'] = [rem_nested_lists(x) for x in mapped20['second_ast']]
    
    # Tokenization
    mapped20['first_tokens'] = [tokenize_task(x) for x in mapped20['third_ast']]
    mapped20['second_tokens'] = [remove_symbols(x) for x in mapped20['first_tokens']]
    mapped20['third_tokens'] = [flatten_list(x) for x in mapped20['second_tokens']]
    
    m20 = mapped20.drop(columns=['initial_ast', 'second_ast', 'first_tokens', 'second_tokens'])
    
    if task_names is not None:
        task_name_tokens = [word_tokenize(token) for token in task_names]
        m20['token_task_names'] = task_name_tokens
    return m20

def change_descriptions(task: dict, task_name:str, module_used: str):
    if task is None:
        return task
    
    mutated_task = mutate_params(normal_module=task, module_used=module_used, task_name=task_name)
    
    return {'task_body': mutated_task['task_body'], 'task_name': mutated_task['task_name']}

def parse_string_to_tasks(string: str):
    """Corrects task's format."""
    string = string.replace('{{ ', '')
    string = string.replace(' }}', '')
    first_parse = string.split(' ')
    second_parse = [item.split('=') for item in first_parse]
    parsed_task = dict()
    for item in second_parse:
        if len(item) >= 2:
            parsed_task.update({item[0]: item[1]})
    return parsed_task

     