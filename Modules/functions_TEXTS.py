#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
import regex as re # type: ignore
import pandas as pd # type: ignore
import json

from FUNCTIONS import load_dic_from_json
from FUNCTIONS import save_dic_to_json

from functions_PARAMETERS import get_physical_units_combined
from functions_PARAMETERS import get_physical_unit_exponent
from functions_PARAMETERS import get_physical_units


#------------------------------
def break_text_in_sentences(text, check_sents = True):
    
    sentence_list = []
    cumulative_index = 0
    
    #print(text + '\n')

    try:
        while True:
            
            text_fraction = text[ cumulative_index : ]
            pattern = 'Soc|[Ff]ig|Suppl|[Ww]t%?|i\.?\s?e|e\.?\s?g|i\.?\s?d|[Nn]os?|[Cc]o|ca|[Ll]td|[Ii]nc|[Rr]ef|[Aa]nal|St|spp?|cf|in'
            match = re.compile(r'(?<![\(\[\s]+({pattern})\s?)[\.\?\!][\)\]]?(?=[\s\,][A-Z0-9\(\[][A-Za-z0-9\-\,\s])'.format(pattern = pattern))
            
            #caso ache a pontuação de final de sentença
            if match.search(text_fraction):
                
                sentece_end_pos = match.search(text_fraction).end()
                sent = text_fraction[ : sentece_end_pos ]
                sentence_list.append( sent )
                cumulative_index += sentece_end_pos + 1

                if check_sents is True:
                    if sent[0] not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                        print('> check sent: ', sent)
                        #time.sleep(0.1)
            
            #caso seja a última sentença
            elif match.search(text_fraction + ' Aa' ):

                sent = text_fraction
                sentence_list.append( sent )                    
                
                if check_sents is True:
                    if sent[0] not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                        print('> check sent: ', sent)
                        #time.sleep(0.1)
                
                break
            
            #caso não haja um ponto final na última sentença
            else:
                break
        
        #print(sentence_list)
        return sentence_list
    
    #caso só tenha uma sentença
    except UnboundLocalError:
        return [text]


#------------------------------
def check_text_language(text, language, word_check_counter = 20):
    
    check_language = False
    
    if language.lower() == 'english':        
        matches = re.findall('\sthe\s', text)
        if len(matches) >= word_check_counter:
            print('Language found: ', language)
            check_language = True
    
    elif language.lower() == 'portuguese':        
        matches = re.findall('\so\s', text)
        if len(matches) >= word_check_counter:
            print('Language found: ', language)
            check_language = True
    
    return check_language


#------------------------------
def check_regex_subs(regex_pattern = None, sub_str = None, filter_name = None, string_text = None, char_range = 30, filter_log_dic = None, filename = ''):
    
    res = re.finditer(regex_pattern, string_text)
    
    for match in res:
        
        before_text = string_text[ match.start() - char_range : match.end() + char_range ]
        after_text = re.sub(regex_pattern, sub_str, before_text)

        if len(before_text) > 2 * char_range:
            
            print('> Filter name: ',  repr(filter_name))
            print('  antes: ', repr(before_text), ' ; depois: ', repr(after_text))

            try:
                filter_log_dic[filter_name]['counts'] += 1
                filter_log_dic[filter_name]['last_print'] = f'{filename}; antes: ' + repr(before_text) + ' ; depois: ' + repr(after_text)

            except KeyError:
                filter_log_dic[filter_name] = {}
                filter_log_dic[filter_name]['counts'] = 1
                filter_log_dic[filter_name]['last_print'] = f'{filename}; antes: ' + repr(before_text) + ' ; depois: ' + repr(after_text)

    #time.sleep(1)
    return filter_log_dic


#------------------------------
def concat_DF_sent_indexes(sent_index, n_sent_to_concat):
    
    sent_index_range = []
    #print('concat sent index: ', sent_index)
        
    #caso o numero de sentença a serem concatenadas seja par        
    if n_sent_to_concat % 2 == 0:            
        delta_index = int(n_sent_to_concat/2)
        #nesse caso, teremos dois sent_index_range a serem testados
        for i in range(delta_index):
            sent_index_range.append( list( range(sent_index - int(n_sent_to_concat/2) + i + 1 , sent_index + int(n_sent_to_concat/2) + i + 1) ) )
            sent_index_range.append( list( range(sent_index - int(n_sent_to_concat/2) - i , sent_index + int(n_sent_to_concat/2) - i) ) )
            
    #caso o numero de sentença a serem concatenadas seja impar
    else:        
        delta_index = int( (n_sent_to_concat - 1)/2)
        for i in range(delta_index):      
            sent_index_range.append( list( range(sent_index - int(n_sent_to_concat/2) + i , sent_index + int(n_sent_to_concat/2) + i + 1) ) )
            sent_index_range.append( list( range(sent_index - int(n_sent_to_concat/2) - i - 1 , sent_index + int(n_sent_to_concat/2) - i) ) )
        sent_index_range.append( list( range(sent_index , sent_index + n_sent_to_concat) ) )
        
    #print('concatenated sent indexes: ', sent_index_range)
    return sent_index_range



#------------------------------
def exist_term_in_string(string = 'string', terms = ['term1', 'term2']):

    found_term = False
    for term in terms:
        term_len = len(term)
        #print('Searching: ', string, '; Term: ', term)
        for char_N in range(len(string)):
            if string[ char_N : char_N + term_len ].lower() == term.lower():
                found_term = True
    
    return found_term



#------------------------------
def filter_chars(string_text, diretorio = None, filename = '', filter_log_dic = None):

    print('Aplicando os filtros de texto...')


    #primeiro filtro - limpando caracteres de quebra de texto e padronizando os traços
    #------------------------------
    pattern_sub_list = [
                        [r'\t{1,50}', '', 'filtro de \t', True, False ],
                        [r'[\ˉ\-\‐\–\−\—\―\─\‑\‒\xad]', '-', 'filtro de traço "-"', True, False ],
                        [r'\n{1,10}[\,\-]\n{1,10}', ' ', 'filtro de "\n-\n"', True, False ],
                        [r'(\-\n){1,50}', ' ', 'filtro de "-\n"', True, False ],
                        [r'\n{1,50}', ' ', 'filtro de "\n"', True, False ],
                        [r'\s{2,50}', ' ', 'filtro de "\s" juntos', True, False ]
                        ]
    string_text, filter_log_dic = filter_and_check(pattern_sub_list, string_text, filter_log_dic, filename)


    #segundo filtro - seleção de caracteres
    #------------------------------

    #abrindo o NotFoundChars.json e o NotFoundChars.txt
    if os.path.exists(diretorio + '/Outputs/NotFoundChars.json'):
        NotFoundChars_dic = load_dic_from_json(diretorio + '/Outputs/NotFoundChars.json')
        NotFoundChars_txt = open(diretorio + '/Outputs/NotFoundChars.txt', 'a', encoding='utf-8')

    else:
        NotFoundChars_dic = {}
        NotFoundChars_dic['chars'] = []
        NotFoundChars_txt = open(diretorio + '/Outputs/NotFoundChars.txt', 'w', encoding='utf-8')
        
    #caracteres que são aceitos
    letter_min = 'abcdefghijklmnopqrstuvwxyz'
    letter_cap = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    numbers = '0123456789' #os numeros em subscrito são aceitos por causa das formulas químicas
    greekletters = 'α βß κ µμ ϵε λƛ ζ δ Φ η ρ π γ ƟθΘ Øø ∂ Δ∆△ ϑ σ τ φ χ ψ ϕ ω ΩΩ Ʃ∑Σξ νυ Υ'
    special_chars = '\n\t'
    pontuation = ' ' + "'" + '"' + '!?;:.·,″“”’′ʼ‘`^_'
    associated_to_numbers = '% -+ ± ×\u2062 ≅≈∼~ ≪≫ <> ⟨⟩ ≤≥≳ Å °'
    associated_to_formulas = '­=≡≠↔∞√∏∫∝ ∣| ∥⊥∇Ψℏ →⇌ • \u2061'
    others = '@#$& *∗ \/(){}[]∧○●◒▲◼⧫◻▽◆Γ̅ '
    accepted_chars = letter_min + letter_cap + numbers + greekletters + special_chars + pontuation + associated_to_numbers + associated_to_formulas + others
    
    #carcateres que não são aceitos
    not_accepted_chars = '□\u2063\u202f\ue5f8\u2009\ue5fb'
    
    new_text = ''
    for char_index in range(len(string_text)):

        #testando os caracters aceitos
        if string_text[char_index] in accepted_chars:
            new_text += string_text[char_index]

        #outros caracteres
        elif string_text[char_index] in not_accepted_chars:
            new_text += ''
        
        elif string_text[char_index] not in not_accepted_chars and string_text[char_index] not in NotFoundChars_dic['chars']:
            #try:
            NotFoundChars_txt.write(filename + ' : ' + string_text[ char_index - 5 : char_index + 5 ] + ' : ' + string_text[char_index] + ' : ' + repr(string_text[char_index]) + '\n' )
            NotFoundChars_dic['chars'].append( string_text[char_index] )
            new_text += ''
            
            #except IndexError:
            #    pass
    
    string_text = new_text
    #fechando os arquivos abertos
    NotFoundChars_txt.close()
    save_dic_to_json(diretorio + '/Outputs/NotFoundChars.json', NotFoundChars_dic)


    #terceiro filtro - substituindo caracteres especiais relacionados a unidades
    #------------------------------
    pattern_sub_list = [
                        [r'[βß]', 'β', 'filtro de "β"', True, False ],
                        [r'[µμ]', 'u', 'filtro de "µ"', True, False ],
                        [r'[ϵε]', 'ϵ', 'filtro de "ϵ"', True, False ],
                        [r'[λƛ]', 'λ', 'filtro de "λ"', True, False ],
                        [r'[ƟΘθ]', 'θ', 'filtro de "Ɵ"', True, False ],
                        [r'[Øø]', 'ø', 'filtro de "ø"', True, False ],
                        [r'[Δ∆△]', 'Δ', 'filtro de "Δ"', True, False ],
                        [r'[ΩΩ]', 'OHM', 'filtro de "OHM"', True, False ],
                        [r'[Ʃ∑Σ]', 'Σ', 'filtro de "Σ"', True, False ],
                        [r'[νυ]', 'υ', 'filtro de "υ"', True, False ],
                        #[r'[\^]', '^', 'filtro de "^"', True, False ],
                        [r'[“”″]', '"', 'filtro de "“"', True, False ],
                        [r'[’′ʼ‘`]', "'", 'filtro de "’"', True, False ],
                        #[r'[\_]', "_", 'filtro de "_', True, False ],
                        [r'[≅≈∼~]', '~', 'filtro de "∼"', True, False ],
                        #[r'[±]', '±', 'filtro de "±"', True, False ],
                        #[r'[\+]', '+', 'filtro de "+"', True, False ],
                        [r'[×\u2062]', '×', 'filtro de "×"', True, False ],
                        [r'[\.·]', '.', 'filtro de "."', True, False ],
                        #[r'[°]', '°', 'filtro de "°"', True, False ],
                        #[r'[Å]', 'Å', 'filtro de "Å"', True, False ],
                        [r'[>⟩]', '>', 'filtro de ">"', True, False ],
                        #[r'[≫]', '≫', 'filtro de "≫"', True, False ],
                        [r'[≥≳]', '≥', 'filtro de "≥"', True, False ],
                        [r'[<⟨]', '<', 'filtro de "<"', True, False ],
                        #[r'[≪]', '≪', 'filtro de "≪"', True, False ],
                        #[r'[≤]', '≤', 'filtro de "≤"', True, False ],
                        #[r'[→]', '→', 'filtro de "→"', True, False ],
                        [r'[*∗]', '*', 'filtro de "*"', True, False ],
                        [r'[∣|]', '|', 'filtro de "|"', True, False ]
                        ]

    string_text, filter_log_dic = filter_and_check(pattern_sub_list, string_text, filter_log_dic, filename)

    #última limpeza
    pattern_sub_list = [[r'(?<=\w)(\s*\,\s*){2,10}(?=\w)', ', ', 'filtro de ", ," juntos', True, False ],
                        [r'(?<=\w)(\s*\.\s*){2,10}(?=\w)', '. ', 'filtro de ". ." juntos', True, False ],
                        [r'(?<=\s+et\.?\sal)\.\s+(?![A-Z])', ' ', 'filtro et al. -> et al"', True, False ],
                        [r'\s{2,50}', ' ', 'filtro de "\s" juntos', True, False ]]
    string_text, filter_log_dic = filter_and_check(pattern_sub_list, string_text, filter_log_dic, filename)


    #print(string_text)
    return string_text, len(string_text), filter_log_dic



#------------------------------
def filter_chemphys_entities(string_text, filename = '', filter_log_dic = None, diretorio = None):

    ## Primeiro se faz uma padronização das unidades físicas para o SI

    #unidades físicas que precisarão ser substituidas no texto
    pattern_sub_list = [
                        [r'(?<=[\s\(\[])>(?=[0-9])', '> ', 'filtro de >5 -> > 5', False, False ],
                        [r'(?<=[\s\(\[0-9])(weight|w|wt|W)\s*/\s*(weight|w|wt|W)(?=[\s\)\]\,\:\.;])', 'wtperc', 'filtro de % (w / w)', False, False ],
                        [r'(?<=[\s\(\[0-9])(weight|w|wt|W)\s*/\s*(vol|v|V)(?=[\s\)\]\,\:\.;])', 'wtvperc', 'filtro de % (w / v)', False, False ],
                        [r'(?<=[\s\(\[0-9])(vol|v|V)\s*/\s*(vol|v|V)(?=[\s\)\]\,\:\.;])', 'volperc', 'filtro de % (v / v)', False, False ],
                        [r'(?<=[\s\(\[0-9])(ppm)(?=[\s\)\]\,\:\.;])', 'mg l-1', 'filtro de ppm -> mg l-1', False, False ],
                        [r'(?<=[\s\(\[0-9])(mu[;\s]+g)(?=[\s\)\]\,\:\.;])', 'ug', 'filtro de mu g -> ug', False, False ],
                        [r'(?<=[\s\(\[0-9])(mu[;\s]+m)(?=[\s\)\]\,\:\.;])', 'um', 'filtro de mu m -> um', False, False ],
                        [r'(?<=[\s\(\[0-9])(cc)(?=[\s\)\]\,\:\.;])', 'cm3', 'cc -> cm3', False, False ],
                        [r'(?<=[\s\(\[0-9])(log\(?10\)?([Cc][Ff][Uu]/m[lL]|[Cc][Ff][Uu])?)(?=[\s\)\]\,\:\.;])', 'log', 'filtro de log(10) -> log', False, False ]
                        ]
    
    string_text, filter_log_dic = filter_and_check(pattern_sub_list, string_text, filter_log_dic, filename)

    #molaridade
    for unit_prefix in ['', 'u','m']:
        pattern_sub_list = [[r'(?<=[0-9]\s?){}M(?=[\s\)\]\,\:\.;])'.format(unit_prefix), 
                             ' {unit_prefix}mol L-1'.format(unit_prefix = unit_prefix),
                             'M -> mol L-1', 
                             False, False ]]
        string_text, filter_log_dic = filter_and_check(pattern_sub_list, string_text, filter_log_dic, filename)

    #coletando as PU combinadas de interesse
    all_PU_combined_dic = []
    for param1, param2 in (("weight", "volume"), ("molarity", "volume"), ("areametric", "weight")):
        PU_combined_dic = get_physical_units_combined(first_parameter = param1, second_parameter = param2, get_inverse = False) 
        all_PU_combined_dic.extend( PU_combined_dic['separated'] )

    for u1, u2 in all_PU_combined_dic:
        
        baseunit1, exponent1 = get_physical_unit_exponent(u1)
        baseunit2, exponent2 = get_physical_unit_exponent(u2)

        def check_rem_exp(unit):
            if unit == '1':
                return ''
            else:
                return unit

        #filtro para padronizar unidades físicas com denominador EX: N/m -> N m-1
        pattern_sub_list = [[r'(?<=[\s\(\[0-9]){baseunit1}\s*\(?{exp1}\)?\s*/\s*{baseunit2}\s*\(?{exp2}\)?(?=[\s\)\]\,\:\.;])'.format(baseunit1 = baseunit1,
                                                                                                                                      exp1 = check_rem_exp(exponent1),
                                                                                                                                      baseunit2 = baseunit2,
                                                                                                                                      exp2 = check_rem_exp(exponent2)), 
                             f'{u1} {baseunit2}-{exponent2}', 
                             f'{u1}/{u2} -> {u1} {baseunit2}-{exponent2}', 
                             False, False ]]
        string_text, filter_log_dic = filter_and_check(pattern_sub_list, string_text, filter_log_dic, filename)
    
        #filtro para padronizar unidades físicas com denominador EX: m(2).g(-1) ou m2g-1 -> m2 g-1
        pattern_sub_list = [[r'(?<=[\s\(\[0-9]){baseunit1}\s*\(?{exp1}\)?[\s\.;]*{baseunit2}\s*\(?-{exponent2}\)?(?=[\s\)\]\,\:\.;])'.format(baseunit1 = baseunit1,
                                                                                                                                             exp1 = check_rem_exp(exponent1),
                                                                                                                                             baseunit2 = baseunit2,
                                                                                                                                             exponent2 = exponent2), 
                             f'{u1} {baseunit2}-{exponent2}', 
                             f'{u1}[.;/\s]{u2} -> {u1} {baseunit2}-{exponent2}', 
                             False, False ]]
        string_text, filter_log_dic = filter_and_check(pattern_sub_list, string_text, filter_log_dic, filename)


    PU_units = get_physical_units()
    #varrer sobre as mais usadas
    for cat in ('areametric', 'distance', 'energy', 'electricpotential', 'log10', 'molarity', 'percentage', 
                'pressure', 'temperature', 'volume', 'weight'):
        
        for unit in PU_units[cat]:
            
            #separar números de unidades físicas. EX: 2% -> 2 %         
            pattern_sub_list = [[r'(?<=[0-9]){unit}(?=[\s\)\]\,\:\.;])'.format(unit = unit), 
                                f' {unit}', 
                                f'separando 0{unit} -> 0 {unit}', 
                                False, False ]]
            string_text, filter_log_dic = filter_and_check(pattern_sub_list, string_text, filter_log_dic, filename)

            #separar intervalos de números ex: 10-20 mg -> 10 - 20 mg
            pattern_sub_list = [[r'(?<=[0-9]\.?[0-9]*\s*({unit})?)\-(?=[0-9]\.?[0-9]*\s*{unit}[\s\)\]\,\:\.;])'.format(unit = unit), 
                                f' - ', 
                                f'separando 10-20 {unit} -> 10 - 20 {unit}', 
                                True, True ]]
            string_text, filter_log_dic = filter_and_check(pattern_sub_list, string_text, filter_log_dic, filename)

    #filtrando nomenclatura química
    #abrindo o dic ~/Inputs/ngrams_to_replace
    ngrams_to_replace_dic = load_dic_from_json(diretorio + '/Inputs/ngrams_to_replace.json')
    for cat in ['inorganic compounds name symbol']:
        
        pattern_sub_list = []
        for k, v in ngrams_to_replace_dic[cat].items():
            pattern_sub_list.append([k, v, f'filtro de {k} -> {v}', False, False ])
        
        string_text, filter_log_dic = filter_and_check(pattern_sub_list, string_text, filter_log_dic, filename)

    #última limpeza de espaços duplicados
    pattern_sub_list = [[r'\s{2,50}', ' ', 'filtro de "\s" juntos', True, False ]]
    string_text, filter_log_dic = filter_and_check(pattern_sub_list, string_text, filter_log_dic, filename)

    #print(string_text)
    #time.sleep(1)
    return string_text, len(string_text), filter_log_dic



#------------------------------
#definingo uma função para substituição de match do REGEX
def filter_and_check(patter_sub_list, string_text, filter_log_dic, filename):
    
    #patter_sub_list é uma lista de items "i", os quais possuem o seguinte formato:        
    #i[0] é pattern do regex
    #i[1] é o string que será usado para a substituição
    #i[2] é o nome do filtro
    #i[3] é o booleno para indicar se o filtro será usado
    #i[4] é o booleno para indicar se a função check será usada
    
    #fazendo as modificações para cada filtro
    for pattern_sub in patter_sub_list:
        
        #caso o filtro seja True
        if (pattern_sub[3] is True):

            #caso o check seja True
            if (pattern_sub[4] is True):
                #print(pattern_sub[0])
                filter_log_dic = check_regex_subs(regex_pattern = pattern_sub[0],
                                                  sub_str = pattern_sub[1],
                                                  filter_name = pattern_sub[2], 
                                                  string_text = string_text,
                                                  filter_log_dic= filter_log_dic,
                                                  filename = filename)
            
            string_text = re.sub(pattern_sub[0], pattern_sub[1], string_text)

    return string_text, filter_log_dic


#------------------------------
def get_term_list_from_TXT(filepath):

    term_list = []
    with open(filepath, 'r', encoding = 'utf-8') as file:
        for line in file.readlines():
            term = line[ : -1] if line[ -1 : ] == '\n' else line[ : ]
            term_list.append(term)
        file.close()
    
    return term_list


#------------------------------
def make_column_text(string_text):
    
    new_text = ''
    for char_index in range(len(string_text)):
        if char_index > 0 and char_index % 100 == 0:
            new_text += '\n'
        new_text += string_text[char_index]
    
    return new_text


#------------------------------
def save_full_text_to_json(text, filename, folder = None, raw_string = False, diretorio=None):

    Dic = {}
    Dic['File_name'] = filename
    if (raw_string is True):
        Dic['Full_text'] = repr(text)
    else:
        Dic['Full_text'] = text
    #caso não haja o diretorio ~/Outputs/folder
    if not os.path.exists(diretorio + '/Outputs/' + folder):
        os.makedirs(diretorio + '/Outputs/' + folder)
    with open(diretorio + '/Outputs/' + folder + '/' + filename + '.json', 'w', enconding='utf-8') as write_file:
        json_str = json.dumps(Dic, sort_keys=True, indent=2)
        write_file.write(json_str)
        write_file.close()
    print('Salvando o raw full text extraido em ~/Outputs/' + folder + '/' + filename + '.json')
    

#------------------------------
def save_sentence_dic_to_csv(sentences_dic, filename, folder = 'Sentences', diretorio=None):
    
    DF = pd.DataFrame([], dtype=object)
    
    #definindo o número da próxima sentença
    min_sent_counter = min( sentences_dic.keys() )
    max_sent_counter = max( sentences_dic.keys() )
    
    for i in range(min_sent_counter, max_sent_counter + 1):

        DF.loc[ i , 'article Number' ] = filename
        DF.loc[ i , 'Sentence' ] = sentences_dic[ i ]['sent']
        DF.loc[ i , 'Section' ] = sentences_dic[ i ]['section']
    
    #caso não haja o diretorio ~/Outputs/folder
    if not os.path.exists(diretorio + f'/Outputs/{folder}'):
        os.makedirs(diretorio + f'/Outputs/{folder}')
    
    DF.to_csv(diretorio + f'/Outputs/{folder}/' + filename + '.csv')
    print(f'Salvando as sentenças extraidas em ~/Outputs/{folder}/' + filename + '.csv')
    
    return max_sent_counter + 1


#------------------------------
def save_sentence_list_to_csv(sentences_generator, filename, documents_counter, folder = None, diretorio=None):
    
    DF = pd.DataFrame([],columns=['article Number', 'Sentence'], dtype=object)
    counter = documents_counter
    for sent in sentences_generator:
        DF.loc[counter] = ( filename, sent )
        counter += 1
    
    #caso não haja o diretorio ~/Outputs/folder
    if not os.path.exists(diretorio + f'/Outputs/{folder}'):
        os.makedirs(diretorio + f'/Outputs/{folder}')

    DF.to_csv(diretorio + f'/Outputs/{folder}/' + filename + '.csv')
    print(f'Salvando as sentenças extraidas em ~/Outputs/{folder}/' + filename + '.csv')
    
    return counter


#------------------------------
def save_text_to_TXT(text, article_file_name, folder= 'Text_not_processed', diretorio=None):

    colunized_text = make_column_text(text)
    
    #print('Salvando o texto extraido...')
    #caso não haja o diretorio ~/Outputs/folder
    if not os.path.exists(diretorio + '/Outputs/' + folder):
        os.makedirs(diretorio + '/Outputs/' + folder)
    with open(diretorio + f'/Outputs/{folder}/' + article_file_name + '_extracted.txt', 'w', encoding='utf-8') as pdf_file_write:
        pdf_file_write.write(colunized_text)
    print(f'Salvando o texto extraido em ~/Outputs/{folder}/' + article_file_name + '_extracted.txt')
    del(colunized_text)