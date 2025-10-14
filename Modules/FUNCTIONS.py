#!/usr/bin/env python3
# -*- coding: utf-8 -*-    

import time
import os
import pandas as pd
import h5py # type: ignore
import json
import numpy as np
import regex as re
from multiprocessing import Pool
from functions_VECTORS import get_vector_from_string
from functions_PARAMETERS import regex_patt_from_parameter
from functions_TOKENS import get_nGrams_list


#------------------------------
def add_text_to_TXT(text, path = None):

    with open(path, 'a', encoding="utf-8") as file:
        file.write(text + '\n')
        file.close()


#------------------------------
def create_article_file_list_with_filename(article_filename):
    
    if (article_filename == 'ATC00000'):
        return []
    else:
        match = re.search('[1-9]{1,5}[0-9]*', article_filename)
        file_number = int(match.group())
        return [ get_tag_name(file_N) for file_N in list(range(file_number + 1)) ]


#------------------------------
def create_h5_matrix(shape = None, filepath = None, dtype=None):

    print('Criando o arquivo h5 em: ', filepath)
    h5_file = h5py.File(filepath, 'w')
    h5_file.create_dataset('data', shape = shape, dtype=dtype)
    h5_matrix = h5_file['data']
    print('H5 matrix created - shape:', h5_matrix.shape)

    return h5_file, h5_matrix


#------------------------------
def error_incompatible_strings_input(input_name, given_input, available_inputs, class_name = ''):
    
    abort_class = False
    
    if given_input.lower() not in available_inputs:
        print(f'Erro para a entrada {input_name}: {given_input}')
        print(f'Selecionar uma entrada adequada para o {input_name} (ver abaixo).')
        print('Entradas disponíveis: ')
        for av_input in available_inputs:
            print(av_input)
        print(f'> Abortando a classe: {class_name}')
        abort_class = True
        
    return abort_class


#------------------------------
def error_print_abort_class(class_name):
        
    print('Erro na instanciação da classe.')
    print(f'> Abortando a classe: {class_name}')


#------------------------------
def extract_inputs_from_csv(csv_filename = '', diretorio = None, mode = 'search_extract'):

    inputs_DF = pd.read_csv(diretorio + f'/Settings/{csv_filename}.csv')
    
    dic = {}
    #varrendo a DF
    if mode.lower() == 'search_extract':
        for line in inputs_DF.index:
            
            dic[line] = {}
            
            #filename
            dic[line]['filename'] = re.findall(r'[A-Za-z0-9].+[A-Za-z0-9]' , inputs_DF.loc[ line , 'filename' ] )[0]

            #index_list to search
            if inputs_DF.loc[ line , 'index_list_name' ].lower() == 'none':
                dic[line]['index_list_name'] = None
            else:
                dic[line]['index_list_name'] = re.findall(r'[A-Za-z0-9].+[A-Za-z0-9]' , inputs_DF.loc[ line , 'index_list_name' ] )[0]
            
            #file type
            dic[line]['file_type'] = re.findall(r'[A-Za-z0-9].+[A-Za-z0-9]' , inputs_DF.loc[ line , 'file_type' ].lower() )[0]
    
            #parameter to extract
            dic[line]['parameter_to_extract'] = inputs_DF.loc[ line , 'parameter_to_extract' ]

            #wholetext or sent
            if str(inputs_DF.loc[ line , 'scan_sent_by_sent']).lower() == 'false':
                dic[line]['scan_sent_by_sent'] = False
            else:
                dic[line]['scan_sent_by_sent'] = True

            #dicionário com os inputs de search-extract
            dic[line]['search_inputs'] = {}

            #filter_unique_results
            if str(inputs_DF.loc[ line , 'filter_unique_results' ]).lower() == 'true':
                dic[line]['search_inputs']['filter_unique_results'] = True            
            elif str(inputs_DF.loc[ line , 'filter_unique_results' ]).lower() == 'false':
                dic[line]['search_inputs']['filter_unique_results'] = False
            else:
                dic[line]['search_inputs']['filter_unique_results'] = False
            
            #literal
            if inputs_DF.loc[ line , 'literal_entry' ].lower() == 'none':
                dic[line]['search_inputs']['literal'] = '()'
            else:
                dic[line]['search_inputs']['literal'] = inputs_DF.loc[ line , 'literal_entry' ]
            
            #semantic
            if inputs_DF.loc[ line , 'semantic_entry' ].lower() == 'none':
                dic[line]['search_inputs']['semantic'] = '()'
            else:
                dic[line]['search_inputs']['semantic'] = inputs_DF.loc[ line , 'semantic_entry' ]

            #lower sentence para procurar termos com similaridade semântica
            if str(inputs_DF.loc[ line , 'search_token_by_token' ]).lower() == 'false':
                dic[line]['search_inputs']['search_token_by_token'] = False
            else:
                dic[line]['search_inputs']['search_token_by_token'] = True

            #lower sentence para procurar
            if str(inputs_DF.loc[ line , 'lower_sentence_for_semantic' ]).lower() == 'false':
                dic[line]['search_inputs']['lower_sentence_for_semantic'] = False
            else:
                dic[line]['search_inputs']['lower_sentence_for_semantic'] = True
            
            #topic
            if inputs_DF.loc[ line , 'lda_sents_topic' ].lower() == 'none':
                dic[line]['search_inputs']['lda_sents_topic'] = None
            else:
                dic[line]['search_inputs']['lda_sents_topic'] = inputs_DF.loc[ line , 'lda_sents_topic' ]

            if inputs_DF.loc[ line , 'lda_articles_topic' ].lower() == 'none':
                dic[line]['search_inputs']['lda_articles_topic'] = None
            else:
                dic[line]['search_inputs']['lda_articles_topic'] = inputs_DF.loc[ line , 'lda_articles_topic' ]
            
            if inputs_DF.loc[ line , 'lsa_sents_topic' ].lower() == 'none':
                dic[line]['search_inputs']['lsa_sents_topic'] = None
            else:
                dic[line]['search_inputs']['lsa_sents_topic'] = inputs_DF.loc[ line , 'lsa_sents_topic' ]

            if inputs_DF.loc[ line , 'lsa_articles_topic' ].lower() == 'none':
                dic[line]['search_inputs']['lsa_articles_topic'] = None
            else:
                dic[line]['search_inputs']['lsa_articles_topic'] = inputs_DF.loc[ line , 'lsa_articles_topic' ]

            if inputs_DF.loc[ line , 'topic_search_mode' ].lower() not in ('cosine', 'major_topics'):
                dic[line]['search_inputs']['topic_search_mode'] = None
            else:
                dic[line]['search_inputs']['topic_search_mode'] = inputs_DF.loc[ line , 'topic_search_mode' ]

            if str( inputs_DF.loc[ line , 'cos_thr' ] ).lower() == 'none':
                dic[line]['search_inputs']['cos_thr'] = None
            else:
                dic[line]['search_inputs']['cos_thr'] = float( inputs_DF.loc[ line , 'cos_thr' ] )
            
            #regex
            if inputs_DF.loc[ line , 'num_param' ].lower() == 'none':
                dic[line]['search_inputs']['regex'] = ''
            else:
                dic[line]['search_inputs']['regex'] = inputs_DF.loc[ line , 'num_param' ]
    
            #filter section
            if inputs_DF.loc[ line , 'filter_section' ].lower() == 'none':
                dic[line]['search_inputs']['filter_section'] = None
            else:    
                if inputs_DF.loc[ line , 'filter_section' ].lower() in ('introduction', 'methodology', 'results'):
                    dic[line]['search_inputs']['filter_section'] = inputs_DF.loc[ line , 'filter_section' ].lower()
                else:
                    print('Erro na linha: ', line, ' ; name entry: ', dic[line]['filename'])
                    print('A entrada de "filter_section" não é compatível.')
                    print('Entradas compatíveis: "introduction", "methodology", "results"')
                    print('Valor de entrada: ', inputs_DF.loc[ line , 'filter_section' ])

            #llm selection
            if inputs_DF.loc[ line , 'llm_model' ].lower() == 'none':
                dic[line]['search_inputs']['llm_model'] = ''
            else:
                dic[line]['search_inputs']['llm_model'] = inputs_DF.loc[ line , 'llm_model' ]

            #status
            for status_input in ('search_status', 'extract_status'):
                
                if str(inputs_DF.loc[ line , status_input ]).lower() != 'finished':
                    dic[line][status_input] = 'ongoing'
                else:    
                    dic[line][status_input] = 'finished'

    
    elif mode.lower() == 'consolidate_df':
        for line in inputs_DF.index:
            
            dic[line] = {}
            
            #filename
            dic[line]['parameter'] = re.findall(r'[A-Za-z0-9].+[A-Za-z0-9]' , inputs_DF.loc[ line , 'parameter' ] )[0]
    
            #parameter to extract
            dic[line]['filenames_to_concatenate'] = inputs_DF.loc[ line , 'filenames_to_concatenate' ]

            #dics to replace terms
            dic[line]['ngrams_to_replace'] = inputs_DF.loc[ line , 'ngrams_to_replace' ]

            #hold_filenames
            if str(inputs_DF.loc[ line , 'hold_filenames' ]).lower() == 'true':
                dic[line]['hold_filenames'] = True
            
            else:
                dic[line]['hold_filenames'] = False
    
            #hold_instances_number
            if str(inputs_DF.loc[ line , 'hold_instances_number' ]).lower() == 'true':
                dic[line]['hold_instances_number'] = True
            
            else:
                dic[line]['hold_instances_number'] = False

            #filter_unique_results
            if str(inputs_DF.loc[ line , 'filter_unique_results' ]).lower() == 'true':
                dic[line]['filter_unique_results'] = True            
            elif str(inputs_DF.loc[ line , 'filter_unique_results' ]).lower() == 'false':
                dic[line]['filter_unique_results'] = False
            else:
                dic[line]['filter_unique_results'] = False

            #type
            dic[line]['parameter_type'] = re.findall(r'[A-Za-z0-9].+[A-Za-z0-9]' , str(inputs_DF.loc[ line , 'type' ]).lower() )[0]

            #match_instances_with_other_parameter
            dic[line]['match_instances_with_other_parameter'] = str(inputs_DF.loc[ line , 'match_instances_with_other_parameter' ])

            #parameters to check correlation with llm
            if str(inputs_DF.loc[ line , 'check_parameter_relations' ]).lower() == 'true':
                dic[line]['check_parameter_relations'] = True
            
            else:
                dic[line]['check_parameter_relations'] = False

            #parameters_to_drop_nan
            dic[line]['parameters_used_to_filter'] = inputs_DF.loc[ line , 'parameters_used_to_filter' ]


    #print(dic)
    return dic


#------------------------------
def filename_gen():
    import random as rdn
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    new_file_name=''
    counter = 0
    while counter < 20:
        index = rdn.randint(0, 61)
        new_file_name += letters[index]
        counter += 1
    return new_file_name


#------------------------------
def generate_PR_results(prediction_list, target_val_list, proba_threshold = 0.5):
        
    #print(prediction_list)
    #print(target_val_list)
    
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    for i in range(len(prediction_list)):
        
        result = prediction_list[i]
        target = target_val_list[i]
        
        if result[1] >= proba_threshold and int(target) == 1:
            true_positives += 1
        elif result[1] >= proba_threshold and int(target) == 0:
            false_positives += 1
        elif result[1] < proba_threshold and int(target) == 1:
            false_negatives += 1
        elif result[1] < proba_threshold and int(target) == 0:
            true_negatives += 1
                
    try:
        precision = true_positives / ( true_positives + false_positives )
    except ZeroDivisionError:
        precision = 0
    try:
        recall = true_positives / ( true_positives + false_negatives )
    except ZeroDivisionError:
        recall = 0
    
    return precision, recall        


#------------------------------
def generate_term_list(terms_in_text_file):

    term_list = []
    for line in terms_in_text_file:
        #print(line)
        term = ''
        for char in line:
            if char == '\n':
                continue
            else:
                term += char
        term_list.append(term)
    
    return term_list


#------------------------------
def get_search_terms_from_input(search_input, diretorio=None):
        
    #definindo o dicionário final da função
    search_input_dic = {}
    
    #--------------------------- termos literais -----------------------
    #carregando os termos do dicionário temporário
    terms = search_input['literal']
    
    #caso seja uma procura via regex
    prim_term_list, sec_term_list, operation_list, found_regex = get_term_list_from_regex_strs(terms)
    
    if found_regex is True:
        
        search_input_dic['literal'] = {}
        search_input_dic['literal']['primary'] = prim_term_list
        search_input_dic['literal']['secondary'] = sec_term_list
        search_input_dic['literal']['operations'] = operation_list
        search_input_dic['literal']['literal_entry'] = terms
    
    else:
        prim_term_list, second_term_list, operation_list = get_term_list_from_tuples_strs(terms)
        
        search_input_dic['literal'] = {}
        search_input_dic['literal']['primary'] = prim_term_list
        search_input_dic['literal']['secondary'] = second_term_list
        search_input_dic['literal']['operations'] = operation_list
        search_input_dic['literal']['literal_entry'] = terms
            

    #--------------------------- termos semânticos -----------------------    
    #encontrando os termos semânticos
    #try:
    search_input_dic['semantic'] = {}

    #carregando os termos do dicionário temporário
    terms = search_input['semantic']

    #encontrando os termos primários
    prim_term_list, second_term_list, operation_list = get_term_list_from_tuples_strs(terms)
    
    #pegando a lista de termos semanticos
    prim_term_list = get_nGrams_list(prim_term_list, diretorio = diretorio)
    sec_term_list = []
    for sec_terms in second_term_list:
        sec_term_list.append(get_nGrams_list(sec_terms, diretorio = diretorio))
        
    search_input_dic['semantic']['primary'] = prim_term_list
    search_input_dic['semantic']['secondary'] = sec_term_list
    search_input_dic['semantic']['operations'] = operation_list
    search_input_dic['semantic']['semantic_entry'] = terms

            
    #--------------------------- padrão regex -----------------------
    #encontrando os padrões regex
    try:
        search_input_dic['regex'] = {}

        #carregando os termos do dicionário temporário
        terms = search_input['regex']
        
        regex_parameter = None
        regex_pattern = None
        if re.search(r'\w+', terms) is not None:
            regex_parameter = re.search(r'\w+', terms).captures()[0]
            #tentando achar o regex pattern do parâmetro introduzido
            regex_pattern = regex_patt_from_parameter(regex_parameter)
        
        if regex_pattern != None:
            regex_entry = regex_parameter
            regex_term = regex_pattern['PU_to_find_regex']
        else:
            regex_entry = ''
            regex_term = ''
            
        search_input_dic['regex']['regex_entry'] = regex_entry
        search_input_dic['regex']['regex_pattern'] = regex_term

    except KeyError:
        search_input_dic['regex']['regex_entry'] = ''
        search_input_dic['regex']['regex_pattern'] = ''
        pass
    
    #print(search_input_dic)
    #time.sleep(10)
    return search_input_dic



#------------------------------
def get_term_list_from_regex_strs(text):
        
    #encontrando os termos primários
    try:
        prim_term = re.search(r'(?<=s\()(.*?)(?=\)e)', text).captures()
        found_regex = True
        
    #caso não haja termo primário
    except AttributeError:
        prim_term = []
        found_regex = False
    
    sec_term = []
    operation = []
    if found_regex is True:

        #encontrando os termos secundários
        sec_terms_find_list = re.findall(r'[\+\-]\s*s\((.*?)(?=\)e)', text)
        for sec_term_str in sec_terms_find_list:
            sec_term.append( [ sec_term_str ] )
            
        #encontrando os operadores
        ops_find_list = re.findall(r'(?<=\)e\s*)[\+\-](?=\s*s\()', text)
        for op in ops_find_list:
            operation.append(op)
    
    return prim_term, sec_term, operation, found_regex



#------------------------------
def get_term_list_from_tuples_strs(text):
    
    prim_terms = []
    sec_terms = []
    operation = []

    if text is not None:

        #termos primários
        try:
            prim_terms = re.findall(r'\b\w+[A-Za-z0-9\_\.\-\:\s]+\w+\b(?=\,)+|\b\w+[A-Za-z0-9\_\.\-\:\s]+\w+\b', re.search(r'(?<=[\s*\(])[A-Za-z0-9\_\.\-\:\,\s]+(?=[\s*\)])', text).group(0) )
        except AttributeError:
            pass

        #termos secundários
        sec_terms_find_list = re.findall(r'[+-]\s*\([A-Za-z0-9\_\.\-\:\,\s]+\)', text)

        for terms_found in sec_terms_find_list:
        
            #pegar o termo para buscar os semanticamente similares
            sec_terms_found = re.findall(r'\b\w+[A-Za-z0-9\_\.\-\:\s]+\w+\b(?=\,)+|\b\w+[A-Za-z0-9\_\.\-\:\s]+\w+\b', re.search(r'(?<=[\s*\(])[A-Za-z0-9\_\.\-\:\,\s]+(?=[\s*\)])', text).group(0) )
            
            #o primeiro char é da operação
            operation.append(terms_found[0])
            sec_terms.append(sec_terms_found)

    return prim_terms, sec_terms, operation



#------------------------------
def get_vectors_from_input(search_input_dic, lsa_dim = 10, lda_dim = 10):

    search_vectors_dic = {}
    search_vectors_dic['topic'] = {}
    search_vectors_dic['topic']['any'] = False

    #encontrando os vetores para procura de tópicos
    search_vectors_dic['topic']['lda_sents'] = get_vector_from_string( search_input_dic['lda_sents_topic'], vector_dim = lda_dim, get_versor = True)
    search_vectors_dic['topic']['lda_articles'] = get_vector_from_string(search_input_dic['lda_articles_topic'], vector_dim = lda_dim, get_versor = True)
    search_vectors_dic['topic']['lsa_sents'] = get_vector_from_string(search_input_dic['lsa_sents_topic'], vector_dim = lsa_dim, get_versor = True)
    search_vectors_dic['topic']['lsa_articles'] = get_vector_from_string(search_input_dic['lsa_articles_topic'], vector_dim = lsa_dim, get_versor = True)

    #caso haja algum vetor
    if type(np.array([])) in (type(search_vectors_dic['topic']['lda_sents']), 
                              type(search_vectors_dic['topic']['lda_articles']),
                              type(search_vectors_dic['topic']['lsa_sents']), 
                              type(search_vectors_dic['topic']['lsa_articles'])):
        
        search_vectors_dic['topic']['any'] = True

    return search_vectors_dic


#------------------------------
def get_file_batch_index_list(total_number, batch_size):
    
    #determinando os slices para os batchs
    print('  Determinando os slices para os batches...')
    slice_indexes = list(range(0, total_number, batch_size))
    batch_indexes = []
    for i in range(len(slice_indexes)):
        try:
            batch_indexes.append([ slice_indexes[i] , slice_indexes[i + 1] - 1])
        except IndexError:
            pass
    batch_indexes.append([slice_indexes[-1] , total_number - 1])
    
    return batch_indexes


#------------------------------
def get_filenames_from_folder(folder, file_type = 'csv', print_msg = False):
    
    try:
        file_list = os.listdir(folder) #lista de arquivos
    except FileNotFoundError:
        print('Erro!')
        print('O diretório não existe:')
        print(folder)
        return
            
    #testar se há arquivos no diretório
    if len(file_list) == 0:
        print('Erro!')
        print('Não há arquivos no diretório:')
        print(folder)
        return
    
    if file_type == 'webscience_csv_report':
        file_type = 'txt'
        
    documents = []
    for filename in file_list:
        if filename[ -len(file_type) : ].lower() == file_type.lower():
            documents.append(filename[ : - ( len(file_type) + 1) ])

    if print_msg is True            :
        print('Procurando arquivos na pasta: ', folder)
        print('Total de arquivos encontrados: ', len(documents))
    
    return sorted(documents)


#------------------------------
def get_sent_from_index(sent_index, diretorio = None):
    
    send_indexes = pd.read_csv(diretorio + '/Outputs/log/sents_index.csv', index_col = 0)
    for article_filename in send_indexes.index:
        initial_sent = send_indexes.loc[article_filename, 'initial_sent']
        last_sent = send_indexes.loc[article_filename, 'last_sent']
        if last_sent >= sent_index >= initial_sent:            
            sent_DF = pd.read_csv(diretorio + f'/Outputs/sents_filtered/{article_filename}.csv', index_col = 0)
            sent = sent_DF.loc[sent_index].values
        else:
            continue
    
    return sent       


#------------------------------
def get_sent_to_predict(token_list, check_sent_regex_pattern = 'z+x?z*'):

    counter_one_char_tokens = 0
    counter_z_char_tokens = 0
    found_regex = False  
        
    #removendo os token listados
    get_sent = True
    for token in token_list:
        temp_token = str(token)
        #primeiro filtro
        #------------------------------
        if re.search(check_sent_regex_pattern, temp_token):
            counter_z_char_tokens += 1
        #segundo filtro
        #------------------------------        
        if len(temp_token) == 1:
            found_regex = True
            counter_one_char_tokens += 1
    
    cond1 = ( counter_z_char_tokens > 2 )
    cond2 = ( counter_one_char_tokens >= 3 )
    cond3 = ( found_regex is False)
    
    if True in (cond1, cond2, cond3):
        get_sent = False
        #print('\n(Filter) Excluindo: ', token_list)
    
    return get_sent


#------------------------------
def get_tag_name(file_N, prefix = 'ATC'):
    if file_N < 10:
        tag = prefix + '00000'
    elif 10 <= file_N < 100:
        tag = prefix + '0000'
    elif 100 <= file_N < 1000:
        tag = prefix + '000'
    elif 1000 <= file_N < 10000:
        tag = prefix + '00'
    elif 10000 <= file_N < 100000:
        tag = prefix + '0'
    elif 100000 <= file_N < 1000000:
        tag = prefix
    return tag + str(file_N)


#------------------------------
def load_h5_matrix(filepath, mode = 'r'):

    print('Carregando o arquivo h5 em: ', filepath)
    h5_file = h5py.File(filepath, mode)
    h5_matrix = h5_file['data']
    print('H5 matrix loaded - shape:', h5_matrix.shape)
    
    return h5_file, h5_matrix
    

#------------------------------
def load_log_info(log_name = None, logpath = None):

    if os.path.exists(logpath):
        try:
            dic = load_dic_from_json(logpath)
            return dic[log_name]
        except KeyError:
            None
    else:
        return None


#------------------------------
def load_dic_from_json(file_path):
    
    with open(file_path, 'r') as file:
        dic = json.load(file)
        file.close()
        
    return dic


#------------------------------
def merge_DFs(DF_filename1, DF_filename2, concatDF_filename, diretorio = None):
    
    DF1 = diretorio + f'/Outputs/{DF_filename1}.csv'
    DF2 = diretorio + f'/Outputs/{DF_filename2}.csv'
    
    DF1 = pd.read_csv(DF1, index_col=[0,1])
    DF2 = pd.read_csv(DF2, index_col=[0,1])
    
    DF = pd.merge(DF1, DF2, on=['Filename', 'Index'], how='outer')
    DF.sort_values(by=['Filename', 'Index'], inplace=True)
    DF.to_csv(diretorio + f'/Outputs/{concatDF_filename}.csv')


#------------------------------
def read_text_from_TXT(path = None):

    with open(path, encoding="utf-8") as file:
        text = file.read()
        file.close()

    return text


#------------------------------
def run_func_in_parallel(func, args, workers = 2):

    with Pool(workers) as pool:
        pool.map(func, args)
        pool.close()
        pool.join()


#------------------------------
def save_dic_to_json(path, dic, sort = True):
    
    #sorting
    keys = list(dic.keys())
    if sort is True:
        keys.sort()
    
    dic = {key: dic[key] for key in keys}

    for key in keys:
        if type(dic[key]) == dict:
            subkeys = list(dic[key].keys())
            if sort is True:
                subkeys.sort()
            subdic = {subkey: dic[key][subkey] for subkey in subkeys}
            dic[key] = subdic

    with open(path, 'w') as file:
        json.dump(dic, file, indent = 3)
        file.close()
    
    print('  salvando o arquivo json em: ', path)


#------------------------------
def saving_acc_to_CSV(last_article_file = 'ATC00000', settings = 'w2vec', acc = 0, folder = '/', diretorio = None):
    
    if not os.path.exists(diretorio + folder + 'wv_accuracy.csv'):
        DF = pd.DataFrame(columns=['Filename', 'Settings', 'Accuracy'])
        DF.set_index(['Filename', 'Settings'], inplace=True)
    else:
        DF = pd.read_csv(diretorio + folder + 'wv_accuracy.csv', index_col = [0,1])
    
    DF.loc[(last_article_file, settings), 'Accuracy'] = acc
    DF.sort_values(by=['Filename', 'Settings'], inplace=True)
    DF.to_csv(diretorio + folder + 'wv_accuracy.csv')


#------------------------------
def update_log(log_names = None, entries = None, logpath = None):

    #confirmando se o número de entradas está batendo
    if len(log_names) == len(entries):

        if not os.path.exists(logpath):
            log_dic = {}            
            for i in range(len(log_names)):
                log_dic[log_names[i]] = entries[i]
        
        else:
            log_dic = load_dic_from_json(logpath)
            for i in range(len(log_names)):
                log_dic[log_names[i]] = entries[i]
        
        save_dic_to_json(logpath, log_dic)
    
    else:
        print('Erro nos inputs da função "update_log"!')
        print('Inserir o mesmo número de elementos nas listas dos args: "log_names" e "entries".')


#------------------------------
def write_text_to_TXT(text, path = None):

    with open(path, 'w', encoding="utf-8") as file:
        file.write(text + '\n')
        file.close()



'''
*** Essas funções abaixo não estão sendo usadas *** 
#------------------------------
#combinações de termo
def find_term_combinations(term):

    terms = []    
    #caso o termo seja composto por várias palavras
    if len(term.split()) > 1:
        concat_term = ''
        #contanando os termos
        for token in term.split():
            concat_term += token
        terms.append(concat_term)            
        for char_index in range(1, len(concat_term)):
            s1_term = concat_term[ : char_index ]  + '-' + concat_term[ char_index : ]
            terms.append(s1_term)
        terms.append(concat_term + '-')

    #caso o termo seja composto só por um token
    else:
        terms.append(term)
        for char_index in range(1, len(term)):
            s1_term = term[ : char_index ]  + '-' + term[ char_index : ]
            terms.append(s1_term)
        terms.append(term + '-')
    
    #print('\n', terms,'\n')    
    return terms


#------------------------------
#combinações das listas de termos
def find_terms_combinations(term_list):
    
    terms = []
    for term_N in range(len(term_list)):
        terms.append(term_list[term_N])
        for char_index in range(1, len(term_list[term_N])):
            s1_term = term_list[term_N][ : char_index ]  + '-' + term_list[term_N][ char_index : ]    
            terms.append(s1_term)
        terms.append(term_list[term_N] + '-')    
    #print('\n',terms,'\n')
    
    return terms
'''
