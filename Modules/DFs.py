#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
import pandas as pd # type: ignore
import numpy as np # type: ignore
import regex as re # type: ignore
import webbrowser as wb

from LLM import llm

from FUNCTIONS import error_incompatible_strings_input
from FUNCTIONS import get_term_list_from_tuples_strs
from FUNCTIONS import load_dic_from_json
from FUNCTIONS import save_dic_to_json

from functions_PARAMETERS import extract_textual_num_parameter_from_json_str
from functions_PARAMETERS import list_numerical_parameter
from functions_PARAMETERS import list_textual_parameter 
from functions_PARAMETERS import list_textual_num_parameter
from functions_PARAMETERS import regex_patt_from_parameter
from functions_PARAMETERS import get_physical_units_converted_to_SI
from functions_PARAMETERS import process_input_parameter

from functions_TOKENS import get_nGrams_list
from functions_TOKENS import get_tokens_from_sent

class DataFrames(object):
    
    def __init__(self, diretorio = None):
        
        print('\n( Class: DataFrames )')

        self.diretorio = diretorio
        self.class_name = 'DataFrames'           



    def set_settings_for_se(self, SE_inputs = None):
        
        print('( Function: set_files )')
        print('Setting Files...')

        self.DF_name = SE_inputs['filename']
        self.file_type = SE_inputs['file_type']
        self.extraction_mode = SE_inputs['extraction_mode']
        self.input_parameter = SE_inputs['parameter_to_extract']
        self.textual_parameter, self.num_parameter = process_input_parameter(self.input_parameter, diretorio = self.diretorio)

        self.lower_sentence_in_textual_search = SE_inputs['search_inputs']['lower_sentence_for_semantic']
        self.search_token_by_token = SE_inputs['search_inputs']['search_token_by_token']
        self.numbers_extraction_mode = SE_inputs['search_inputs']['numbers_extraction_mode']
        self.filter_unique_results = SE_inputs['search_inputs']['filter_unique_results']

        #abrindo o dic ~/Inputs/ngrams_to_replace
        self.replace_ngrams = False
        self.ngrams_to_replace = load_dic_from_json(self.diretorio + '/Inputs/ngrams_to_replace.json')
        try:
            self.ngrams_to_replace = self.ngrams_to_replace[self.textual_parameter]
            self.replace_ngrams = True
            print(f'\nInput ({self.textual_parameter}) encontrado no dicionário "~/Inputs/ngrams_to_replace"')
        
        except KeyError:
            print(f'\nInput ({self.textual_parameter}) não encontrado no dicionário "~/Inputs/ngrams_to_replace"')



    def extract_textual_parameters(self, text: str, text_index: int, parameter: str, term_list_to_be_found: list):

        
        self.parameters_extracted[text_index]['textual_params'] = {}
        self.parameters_extracted[text_index]['textual_params']['text_index'] = text_index
        self.parameters_extracted[text_index]['textual_params']['parameter'] = parameter
        self.parameters_extracted[text_index]['textual_params']['got_parameter_from_sent'] = False

        terms_list = []
        #varrendo os termos
        for term in term_list_to_be_found:
            
            mod_text = text
            if self.lower_sentence_in_textual_search is True:
                mod_text = text.lower()

            #caso seja varrido token por token
            if self.search_token_by_token is True:

                tokens_raw_text = get_tokens_from_sent(text)
                tokens_mod_text = get_tokens_from_sent(mod_text)

                for i in range(len(tokens_mod_text)):
                    if tokens_mod_text[i] == term:

                        original_token = tokens_raw_text[i]

                        #substituindo o termo caso ele esteja no ~/Inputs/ngrams_to_replace.json
                        if self.replace_ngrams is True:
                            try:
                                original_token = self.ngrams_to_replace[original_token]
                            
                            except KeyError:
                                pass                    
                        
                        #adicionando o termo na lista
                        terms_list.append(original_token)

            #coletando o termo
            elif re.finditer(term, mod_text):
                
                matches = re.finditer(term, mod_text)
                
                for match in matches:

                    original_ngram = text[ match.span()[0] : match.span()[1] ]

                    #substituindo o termo caso ele esteja no ~/Inputs/ngrams_to_replace.json
                    if self.replace_ngrams is True:
                        try:
                            original_ngram = self.ngrams_to_replace[original_ngram]
                        
                        except KeyError:
                            pass
                    
                    #adicionando o termo na lista
                    terms_list.append(original_ngram)
        
        terms_list.sort()
        self.parameters_extracted[text_index]['textual_params']['all_textual_output'] = terms_list
        
        #somando para o total de textual_params extraidos
        self.parameters_extracted['total_textual_outputs_extracted'] += len( self.parameters_extracted[text_index]['textual_params']['all_textual_output'] )



    def extract_textual_numerical_parameters(self, text, text_index, parameter):
        
        print('extract_textual_numerical_parameters')
        
        #filtro de LLM
        llm_model = llm('mistral-large:123b-instruct-2407-q2_K')
        text = llm_model.extract_textual_num_parameter(parameter, text)

        textual_params, num_params = extract_textual_num_parameter_from_json_str(text)
        print(textual_params)
        print(num_params)
        



    def extract_numerical_parameters(self, text: str, text_index: int, parameter: str, extract_mode: str = 'all'):

        #filtro de LLM
        llm_model = llm('mistral-large:123b-instruct-2407-q2_K')
        text = llm_model.extract_num_parameter(parameter, text)

        #obtendo os padrões regex para encontrar e para não encontrar
        parameters_patterns = regex_patt_from_parameter(parameter)
        
        #print('\nPU_to_find_regex:\n', parameters_patterns['PU_to_find_regex'])

        #chaves do dicionário
        self.parameters_extracted[text_index]['num_params'] = {}
        self.parameters_extracted[text_index]['num_params']['PUs_extracted'] = None
        self.parameters_extracted[text_index]['num_params']['parameter'] = parameter
        self.parameters_extracted[text_index]['num_params']['extract_error'] = False
        self.parameters_extracted[text_index]['num_params']['all_num_output'] = []
        self.parameters_extracted[text_index]['num_params']['got_parameter_from_sent'] = False
        
        #encontrandos os parâmetros numéricos
        if re.finditer(parameters_patterns['PU_to_find_regex'], text ):
            
            match_n = 1
            for match in re.finditer(parameters_patterns['PU_to_find_regex'], text ):

                PU_found = match.groups()[-2]

                #coletando o PU
                self.parameters_extracted[text_index]['num_params'][match_n] = {}
                self.parameters_extracted[text_index]['num_params'][match_n]['PU_not_converted'] = PU_found.split()
                self.parameters_extracted[text_index]['num_params'][match_n]['num_not_converted'] = []
                
                for find in match.groups():
                    
                    #o padrão regex é para pegar as entradas que só tenham números (sem outros caracteres)
                    if (find is not None) and (find not in self.parameters_extracted[text_index]['num_params'][match_n].values() ):

                        #checando se o parâmetro é numérico ou a unidade PU
                        try:
                            rfind = re.sub(r'\s', '', find)
                            self.parameters_extracted[text_index]['num_params'][match_n]['num_not_converted'].append( float(rfind) )
                        
                        except ValueError:
                            continue

                match_n += 1
        
        #varrendo os findings
        for match_n in [ key for key in self.parameters_extracted[text_index]['num_params'].keys() if type(key) == int ]:
            
            #encontrando as entradas numéricas com os números e as PUs
            try:                
                if match_n > 1 and extract_mode == 'one':
                    self.parameters_extracted[text_index]['num_params']['extract_error'] = True
                    break

                #fazendo a conversão das unidades
                factor_to_multiply , factor_to_add , PU_in_SI = get_physical_units_converted_to_SI( self.parameters_extracted[text_index]['num_params'][match_n]['PU_not_converted'] )
                
                if None not in (factor_to_multiply , factor_to_add , PU_in_SI):
                    
                    #a unidade PU no formato SI se pega uma vez por sentença
                    if self.parameters_extracted[text_index]['num_params']['PUs_extracted'] is None:
                        self.parameters_extracted[text_index]['num_params']['PUs_extracted'] = PU_in_SI
                    
                    for num_not_converted in self.parameters_extracted[text_index]['num_params'][match_n]['num_not_converted']:
                        self.parameters_extracted[text_index]['num_params']['all_num_output'].append( round( ( float(num_not_converted) * factor_to_multiply) + factor_to_add , 9) )
                
                else:
                    print('ERRO de conversão das PUs. Checar se as unidades da PU: ', self.parameters_extracted[text_index]['num_params'][match_n]['PU_not_converted'], 
                          ' estão na função "get_physical_units_converted_to_SI".')
                    time.sleep(5)


            except ValueError:
                pass

            del self.parameters_extracted[text_index]['num_params'][match_n]
        
        #somando para o total de num_params extraidos
        self.parameters_extracted['total_num_outputs_extracted'] += len( self.parameters_extracted[text_index]['num_params']['all_num_output'] )



    def extract_vals_from_sent(self, sent: str, sent_index: int, term_list_to_be_found: list):

        #só entramos na sequência abaixo para coletar os parâmetros
        if self.extraction_mode != 'select_sentences':

            found_num_parameter = False
            found_textual_parameter = False

            #checando se o parâmetro irá para a extração numérica
            if self.num_parameter in list_numerical_parameter():
                
                found_num_parameter = True

                #extraindo os parâmetros numéricos com as unidades físicas
                self.extract_numerical_parameters(sent, sent_index, self.num_parameter, extract_mode = self.numbers_extraction_mode)
                
                #caso tenha sido extraído algum output numérico corretamente
                if len(self.parameters_extracted[sent_index]['num_params']['all_num_output']) > 0 and self.parameters_extracted[sent_index]['num_params']['extract_error'] is False:
                    
                    print('> Extracted numerical outputs - n_num_outputs: ', len(self.parameters_extracted[sent_index]['num_params']['all_num_output']), ' ; SI_units: ', self.parameters_extracted[sent_index]['num_params']['PUs_extracted'] )
                    print('>', self.parameters_extracted[sent_index]['num_params']['all_num_output'] )

                    #adicionando as SI PUs que serão exportadas para a consolidated_DF
                    self.SI_PUs_dic_to_record[self.num_parameter] = self.parameters_extracted[sent_index]['num_params']['PUs_extracted']


            #checando se o parâmetro irá para a extração textual                                                
            if self.textual_parameter in list_textual_parameter(diretorio=self.diretorio) or len(term_list_to_be_found) > 0:

                found_textual_parameter = True

                #extraindo os parâmetros textuais
                self.extract_textual_parameters(sent, sent_index, self.textual_parameter, term_list_to_be_found)
                
                #caso tenha sido extraído algum output textual
                if len(self.parameters_extracted[sent_index]['textual_params']['all_textual_output']) > 0:
                    print('> Extracted textual outputs - n_textual_outputs: ', len(self.parameters_extracted[sent_index]['textual_params']['all_textual_output']), ' )')
                    print('> Parâmetros textuais extraídos: ', self.textual_parameter)
                    print('> ', self.parameters_extracted[sent_index]['textual_params']['all_textual_output'] )

        
            #checando se o parâmetro irá para a extração textual e numérica junto
            if self.input_parameter in list_textual_num_parameter():
                
                found_num_parameter = True
                found_textual_parameter = True
                result = self.extract_textual_numerical_parameters(sent, sent_index, self.input_parameter) #XXXXXX mexer nisso


            #caso o parâmetro introduzido não esteja na lista do functions_PARAMETERS
            if True not in (found_num_parameter, found_textual_parameter):
                
                #listar os parâmetros disponíveis para extração
                available_inputs = list_numerical_parameter() + list_textual_num_parameter() + list_textual_parameter(diretorio=self.diretorio)
                abort_class = error_incompatible_strings_input('input_parameter', self.input_parameter, available_inputs, class_name = self.class_name)
                if abort_class is True:
                    return

            #caso o método de coleta seja manual
            if self.extraction_mode == 'collect_parameters_manual':
                
                while True:
                    
                    user_entry = str(input('\nConfirme o(s) valor(es) extraidos (yes/y): '))

                    if user_entry.lower() in ('y', 'yes'):
                        try:
                            if len(self.parameters_extracted[sent_index]['num_params']['all_num_output']) > 0:
                                self.parameters_extracted[sent_index]['num_params']['got_parameter_from_sent'] = True
                        except KeyError:
                            pass
                        
                        try:
                            if len(self.parameters_extracted[sent_index]['textual_params']['all_textual_output']) > 0:
                                self.parameters_extracted[sent_index]['textual_params']['got_parameter_from_sent'] = True
                        except KeyError:
                            pass
                        
                        break
                    
                    elif user_entry == 'exit':
                        print('> Abortando função: DataFrames.get_data')
                        return
                    
                    elif user_entry.lower() in ('n', 'no'):
                        break

                    else:
                        print('ERRO > Input inválido (digite "yes", "no" ou "exit")\n')
                
            elif self.extraction_mode == 'collect_parameters_automatic':
                
                try:
                    if len(self.parameters_extracted[sent_index]['num_params']['all_num_output']) > 0:
                        self.parameters_extracted[sent_index]['num_params']['got_parameter_from_sent'] = True
                except KeyError:
                    pass
                
                try:
                    if len(self.parameters_extracted[sent_index]['textual_params']['all_textual_output']) > 0:
                        self.parameters_extracted[sent_index]['textual_params']['got_parameter_from_sent'] = True
                except KeyError:
                    pass



    def generate_search_report(self):
        
        #abrindo o SE report            
        if os.path.exists(self.diretorio + '/Settings/SE_inputs.csv'):
            search_report_DF = pd.read_csv(self.diretorio + '/Settings/SE_inputs.csv', index_col = 0)

            search_report_DF.loc[self.DF_name, 'total_extracted'] = self.search_report_dic['export'][self.DF_name]['total_finds']
            search_report_DF.loc[self.DF_name, 'articles_extracted'] = self.search_report_dic['export'][self.DF_name]['article_finds']
            search_report_DF.loc[self.DF_name , 'export_status' ] = 'finished'

            search_report_DF.sort_index(inplace=True)
            search_report_DF.to_csv(self.diretorio + '/Settings/SE_inputs.csv')
            print('Salvando o SE report em ~/Settings/SE_inputs.csv')

            save_dic_to_json(self.diretorio + '/Outputs/dataframes/SI_PUs.json', self.SI_PUs_dic_to_record)
            print('Salvando o PUs SI em ~/Outputs/dataframes/SI_PUs.json')



    def get_data(self):

        print('( Function: get_data )')

        #dic para salvar as SI PUs padronizadas para cada features (ou parâmetros)
        if not os.path.exists(self.diretorio + '/Outputs/dataframes/SI_PUs.json'):
            self.SI_PUs_dic_to_record = {}
        else:
            self.SI_PUs_dic_to_record = load_dic_from_json(self.diretorio + '/Outputs/dataframes/SI_PUs.json')

        #coletando os termos a serem encontrados na procura textual
        term_list_to_be_found = get_nGrams_list( [ self.textual_parameter ], diretorio=self.diretorio)

        #caso nao haja no ner_rules usa-se a entrada manual introduzida no SE_inputs.csv
        if term_list_to_be_found is None:
            term_list_to_be_found, sec_terms, operation = get_term_list_from_tuples_strs(self.textual_parameter)

        #checando se existe um DF de fragmentos
        print(f'\nProcurando... {self.diretorio}/Outputs/extracted/' + f'{self.DF_name}.csv')
        if os.path.exists(self.diretorio + '/Outputs/extracted/' + f'{self.DF_name}.csv'):
            
            self.extracted_sents_DF = pd.read_csv(self.diretorio + '/Outputs/extracted/' + f'{self.DF_name}.csv', index_col=[0,1], dtype=object)
            articles_filenames = list(np.unique(self.extracted_sents_DF.index.get_level_values(0).values))
            print(f'Carregando o DataFrame com os textos extraidos (~/Outputs/extracted/{self.DF_name}.csv)')

            #abrindo o search-extract report
            if os.path.exists(self.diretorio + f'/Outputs/log/se_report.json'):
                
                #carregando o dicionário
                self.search_report_dic = load_dic_from_json(self.diretorio + f'/Outputs/log/se_report.json')
                
                if self.search_report_dic['search'][self.DF_name]['searching_status'] != 'finished':
                    print(f'Erro! O processo de extração para o search_input {self.DF_name} ainda não terminou.' )
                    return

                try:
                    self.search_report_dic['export']

                    try:
                        #último filename processado
                        last_article_processed = self.search_report_dic['export'][self.DF_name]['last_article_processed']
                        last_article_index = articles_filenames.index(last_article_processed) + 1
                        print('Last file searched: ', last_article_processed)

                    except KeyError:
                        self.search_report_dic['export'][self.DF_name] = {}
                        self.search_report_dic['export'][self.DF_name]['last_article_processed'] = None
                        self.search_report_dic['export'][self.DF_name]['total_finds'] = 0
                        self.search_report_dic['export'][self.DF_name]['article_finds'] = 0
                        last_article_index = 0
                
                
                except KeyError:
                    self.search_report_dic['export'] = {}
                    self.search_report_dic['export'][self.DF_name] = {}
                    self.search_report_dic['export'][self.DF_name]['last_article_processed'] = None
                    self.search_report_dic['export'][self.DF_name]['total_finds'] = 0
                    self.search_report_dic['export'][self.DF_name]['article_finds'] = 0
                    last_article_index = 0                

            else:
                print('Erro! LOG counter_se_report não encontrado em ~/outputs/log' )
                print(f'Erro! O processo de extração para o search_input {self.DF_name} não foi feito.' )
                return

            for article_filename in articles_filenames[ last_article_index :  ]:
                    
                print(f'\n------------------------------------------')
                print(f'Extracting parameters from {article_filename}')
                
                #dicionário para coletar os parâmetros numéricos extraídos
                self.parameters_extracted = {}
                self.parameters_extracted['filename'] = article_filename
                self.parameters_extracted['total_textual_outputs_extracted'] = 0
                self.parameters_extracted['total_num_outputs_extracted'] = 0

                print('\nParameter: ( ', self.input_parameter, ' )') 
                print('Fragments extracted from: ', article_filename)

                #varrendo as linhas com as sentenças para esse artigo (input_DF)
                for i in self.extracted_sents_DF.loc[ (article_filename , ) , : ].index:

                    #sentença
                    sent = self.extracted_sents_DF.loc[ (article_filename , i ) , self.DF_name ]
                    
                    #index de sentença
                    sent_index = int( self.extracted_sents_DF.loc[ (article_filename , i ) , self.DF_name + '_index' ] )

                    #criando uma key para o sent_index
                    self.parameters_extracted[sent_index] = {}
                    self.parameters_extracted[sent_index]['sent'] = sent
                                                            
                    #Mostrando as sentenças a serem processadas para esse artigo
                    print(f'\nIndex {i} (sent_index {sent_index}):', sent, '\n')
                    
                    #extraindo os valores da sentença
                    self.extract_vals_from_sent(sent, sent_index, term_list_to_be_found)
                                                        
                
                while True: 
                    try:
                        #escolhendo entre modo de seleção de sentenças ou modo de coleta de parâmetros  
                        if self.extraction_mode == 'select_sentences':
                            print('\nDigite o(s) index(es) das sentenças de interesse (digite valores inteiros)')
                            print('Outros comandos: "+" para ignorar esse artigo e ir para o próximo; "open" para abrir o artigo; e "exit" para sair.')
                            user_input_val = str(input('Index: '))
                            
                        elif self.extraction_mode in ('collect_parameters_automatic', 'collect_parameters_manual'):
                            user_input_val = '*'
                        
                        #processando o input                                        
                        if user_input_val.lower() == 'open':
                            wb.open_new(self.diretorio + '/DB/' + article_filename + f'.{self.file_type}')
                            continue

                        elif user_input_val.lower() == '*':
                            break
                        
                        elif user_input_val.lower() == '+':
                            break
                        
                        elif user_input_val.lower() == 'exit':
                            print('> Abortando função: DataFrames.get_data')
                            return
                        
                        else:
                            #caso o modo seja para coletar as sentenças para treinamento ML                                                                                
                            if self.extraction_mode == 'select_sentences':
                                self.parameters_extracted['selected_sent_index'] = None
                                try:
                                    #verificando se esse parametro inserido é relativo ao valor de key das sentenças coletadas
                                    self.parameters_extracted['selected_sent_index'] = int(user_input_val)
                                    break
                                except (TypeError, KeyError):
                                    print('Erro! Inserir um index válido para coletar a sentença.')
                                    continue                                                                                            
                
                    except ValueError:
                        print('--------------------------------------------')
                        print('Erro!')
                        print('Inserir valor válido')
                        break                                                                                        
        
                
                #se o último input introduzido não foi "+" (que é usado para passar para o proximo artigo)   
                if user_input_val != '+':
                                    
                    #caso seja modo de coleta de dados e algum parametro foi extraído
                    if self.extraction_mode != 'select_sentences':
                        
                        #print(self.parameters_extracted)

                        #definindo um dicionário para colocar os outputs consolidados
                        self.parameters_extracted['parameters_extracted'] = {}

                        #definindo uma lista para colocar todos os parâmetros coletados
                        all_values_collected_list = []

                        #caso todos os parâmetros numéricos entrem
                        if self.parameters_extracted['total_num_outputs_extracted'] > 0:

                            self.parameters_extracted['parameters_extracted'][self.num_parameter] = {}
                            self.parameters_extracted['parameters_extracted'][self.num_parameter]['outputs'] = []
                            self.parameters_extracted['parameters_extracted'][self.num_parameter]['extracted_sent_index'] = []

                            #varrendo todas as sentenças que tiveram dados extraidos
                            clustered_numbers = ''
                            for sent_index in [ i for i in self.parameters_extracted.keys() if type(i) == int ]:
                                
                                #caso os parametros não foram coletados na sentença (isso acontece somente no modo de coleta manual)
                                if self.parameters_extracted[sent_index]['num_params']['got_parameter_from_sent'] is False:
                                    continue
                                
                                #varrendo os outputs no formato: [ counter , sent_index , val ]
                                for val in self.parameters_extracted[sent_index]['num_params']['all_num_output']:
                                    
                                    if self.filter_unique_results == True:
                                        if val not in all_values_collected_list:
                                            clustered_numbers += str( round(float( val ), 10) ) + ', '
                                            all_values_collected_list.append(val)
                                        
                                    else:
                                        clustered_numbers += str( round(float( val ), 10) ) + ', '
                                    
                                    if sent_index not in self.parameters_extracted['parameters_extracted'][self.num_parameter]['extracted_sent_index']:
                                        self.parameters_extracted['parameters_extracted'][self.num_parameter]['extracted_sent_index'].append(sent_index)
                            
                            if len(clustered_numbers) > 0:
                                
                                #tirando o último ', '
                                clustered_numbers = clustered_numbers[ : -2]
                                self.parameters_extracted['parameters_extracted'][self.num_parameter]['outputs'].append( clustered_numbers )
                                
                        #caso todos os parâmetros textuais entrem
                        if self.parameters_extracted['total_textual_outputs_extracted'] > 0:

                            self.parameters_extracted['parameters_extracted'][self.textual_parameter] = {}
                            self.parameters_extracted['parameters_extracted'][self.textual_parameter]['outputs'] = []
                            self.parameters_extracted['parameters_extracted'][self.textual_parameter]['extracted_sent_index'] = []

                            #varrendo todas as sentenças que tiveram dados extraidos
                            for sent_index in [ i for i in self.parameters_extracted.keys() if type(i) == int ]:

                                #caso os parametros não foram coletados na sentença (isso acontece somente no modo de coleta manual)
                                if self.parameters_extracted[sent_index]['textual_params']['got_parameter_from_sent'] is False:
                                    continue

                                #varrendo os outputs no formato: [ counter , sent_index , val ]
                                for val in self.parameters_extracted[sent_index]['textual_params']['all_textual_output']:
                                    
                                    #caso os parâmetros sejam extraidos somente um vez
                                    if self.filter_unique_results == True:
                                        if val not in self.parameters_extracted['parameters_extracted'][self.textual_parameter]['outputs']:
                                            
                                            self.parameters_extracted['parameters_extracted'][self.textual_parameter]['outputs'].append( val )
                                    
                                            if sent_index not in self.parameters_extracted['parameters_extracted'][self.textual_parameter]['extracted_sent_index']:
                                                self.parameters_extracted['parameters_extracted'][self.textual_parameter]['extracted_sent_index'].append(sent_index)
                                    
                                    #caso todos os parâmetros textuais de todas as sentenças sejam extraídos
                                    else:
                                        self.parameters_extracted['parameters_extracted'][self.textual_parameter]['outputs'].append( val )
                                        
                                        if sent_index not in self.parameters_extracted['parameters_extracted'][self.textual_parameter]['extracted_sent_index']:
                                            self.parameters_extracted['parameters_extracted'][self.textual_parameter]['extracted_sent_index'].append(sent_index)

                        #apagando as keys de cada sentença
                        for sent_index in [ i for i in self.parameters_extracted.keys() if type(i) == int ]:
                            del(self.parameters_extracted[sent_index])

                        #gerando a DF com os dados colletados
                        print('\nSummary: parameters extracted from file: ', article_filename)
                        print(self.parameters_extracted)
                        #time.sleep(5)

                        #checando se algum parâmetro foi coletado
                        got_num_param = False
                        got_textual_param = False
                        try:
                            got_num_param = len(self.parameters_extracted['parameters_extracted'][self.num_parameter]['outputs']) > 0 
                        except KeyError:
                            pass
                        try:
                            got_textual_param = len(self.parameters_extracted['parameters_extracted'][self.textual_parameter]['outputs']) > 0
                        except KeyError:
                            pass
                        
                        if True in (got_num_param, got_textual_param):
                            self.export_to_DF()


            #consolidando o report na DF caso seja o ultimo arquivo de procura        
            if article_filename == self.extracted_sents_DF.index.levels[0][-1]:
                self.generate_search_report()


        else:
            print('Erro! Não foi encontrado um DF de fragamentos de artigos.')
            print('> Abortando a classe: DataFrames')
            return



    def export_to_DF(self):

        print('\nExporting to DF...')

        if not os.path.exists(self.diretorio + '/Outputs/dataframes'):
            os.makedirs(self.diretorio + '/Outputs/dataframes')

        #trabalhando no output_DF            
        for param in (self.textual_parameter, self.num_parameter):

            if param is not None:

                #checando se já existe um output Data Frame para esses parâmetros
                if os.path.exists(self.diretorio + f'/Outputs/dataframes/{self.DF_name}_{param}.csv'):
                    
                    self.output_DF = pd.read_csv(self.diretorio + f'/Outputs/dataframes/{self.DF_name}_{param}.csv', index_col=[0,1], dtype=object)
                    self.output_DF.index.names = ['Filename', 'Counter']
                    #print(f'Carregando o DataFrame de OUTPUT (~/Outputs/extracted/{self.DF_name}_{param}.csv)')
                
                else:
                    print(f'Output DF {self.DF_name}_{param}.csv não encontrado.')
                    print(f'Criando o output_DF data frame: {self.DF_name}_{param}.csv')
                    
                    #caso tenha que ser gerada a output_DF
                    self.output_DF = pd.DataFrame(columns=['Filename', 'Counter'], dtype=object)
                    self.output_DF.set_index(['Filename', 'Counter'], inplace=True)
                
                for i in range( len(self.parameters_extracted['parameters_extracted'][param]['outputs']) ):
                    
                    #article_name
                    article_name = self.parameters_extracted['filename']
                
                    #salvando no output_DF
                    output_val = self.parameters_extracted['parameters_extracted'][param]['outputs'][i]
                    
                    self.output_DF.loc[ ( article_name , i ) , param ] = output_val
                    self.output_DF.loc[ ( article_name , i ) , param + '_index' ] = str( self.parameters_extracted['parameters_extracted'][param]['extracted_sent_index'] )
        
                #salvando a output_DF
                self.output_DF.to_csv(self.diretorio + f'/Outputs/dataframes/{self.DF_name}_{param}.csv')
                print(f'\nSalvando output_DF para {article_name}...')

                #contadores para exportar nas DFs de report
                self.search_report_dic['export'][self.DF_name]['last_article_processed'] = self.parameters_extracted['filename']
                self.search_report_dic['export'][self.DF_name]['total_finds'] += len(self.parameters_extracted['parameters_extracted'][param]['outputs'])
                self.search_report_dic['export'][self.DF_name]['article_finds'] += 1
                
                #salvando o DF report
                save_dic_to_json(self.diretorio + f'/Outputs/log/se_report.json', self.search_report_dic)



    def set_settings_to_concatenate(self, dic_inputs = None):
        
        self.parameter_to_consolidate = dic_inputs['parameter']
        self.filenames_to_concatenate = dic_inputs['filenames_to_concatenate']
        self.hold_filenames = dic_inputs['hold_filenames']
        self.hold_instances_number = dic_inputs['hold_instances_number']
        self.parameter_type = dic_inputs['parameter_type']
        self.match_instances_with_other_parameter = dic_inputs['match_instances_with_other_parameter']



    def consolidate_DF(self):

        print('( Function: consolidate_DF )')

        print(f'\nProcurando... {self.diretorio}/Outputs/dataframes/_consolidated_DF.csv')
        if not os.path.exists(self.diretorio + f'/Outputs/dataframes/_consolidated_DF.csv'):
            print(f'Criando a DF consolidada... (~/Outputs/dataframes/_consolidated_DF.csv)')
            self.consolidated_DF = pd.DataFrame(columns=[self.parameter_to_consolidate], index=[[],[]], dtype=object)
            self.consolidated_DF.index.names = ['Filename', 'Counter']
            self.consolidated_DF.to_csv(self.diretorio + f'/Outputs/dataframes/_consolidated_DF.csv')

        #carregando a DF consolidada
        else:
            print(f'Abrindo a consolidated DF: {self.diretorio}/Outputs/dataframes/_consolidated_DF.csv')
            self.consolidated_DF = pd.read_csv(self.diretorio + f'/Outputs/dataframes/_consolidated_DF.csv', index_col=[0,1], dtype=object)
            #adicionando a coluna com o parametro atual a ser extraido
            print(self.parameter_to_consolidate, self.consolidated_DF.columns)
            if self.parameter_to_consolidate not in self.consolidated_DF.columns:
                self.consolidated_DF[self.parameter_to_consolidate] = np.nan


        #concatendo os DFs
        concat_extracted_DFs = self.concat_extracted_DFs()

        if self.consolidated_files[self.parameter_to_consolidate] != 'finished':

            #filenames para varrer
            articles_filenames = list(np.unique(concat_extracted_DFs.index.get_level_values(0).values))
            
            for articles_filename in articles_filenames:
                    
                print(f'\n------------------------------------------')
                print(f'Extracting parameters from {articles_filename}')

                new_vals = []
                for entry in concat_extracted_DFs.loc[ (articles_filename , ), self.parameter_to_consolidate ].values:

                    #se forem valores números, junta-se tudo em uma única string
                    if self.parameter_type == 'numerical':
                        vals = re.findall(r'\-?[0-9][\.0-9]*e?\-?[0-9]*', entry)
                        if len(vals) > 0:
                            for i in vals:
                                if float(i) not in new_vals:
                                    new_vals.append(float(i))
                            
                            str_vals = ''
                            for i in new_vals:
                                str_vals += str(i) + ', '
                            new_vals = [str_vals[ : -2]]
                    
                    #caso sejam textuais
                    elif self.parameter_type == 'textual':
                        new_vals.append(entry)

                self.number_new_instances_extracted = len(new_vals)
                
                #coletando o atual número de instancias para o articles_filenames
                if self.check_conditions_to_consolidated_DF(articles_filename) is True:

                    counter = 0
                    #caso o articles_filenames não esteja na DF consolidada
                    if articles_filename not in np.unique(self.consolidated_DF.index.get_level_values(0).values) and self.hold_instances_number is False:
                        for new_val in new_vals:
                            self.consolidated_DF.loc[ ( articles_filename , counter ) , self.parameter_to_consolidate ] = new_val
                            counter += 1

                    else:

                        #enquanto tiver só uma coluna
                        if len(self.consolidated_DF.columns) == 1:
                            for new_val in new_vals:
                                self.consolidated_DF.loc[ ( articles_filename , counter ) , self.parameter_to_consolidate ] = new_val
                                counter += 1

                        #caso já existam entradas na consolidated DF                    
                        else:
                            entry_dic = {}

                            #fazendo as combinações dos inputs                    
                            consolidated_DF_copy = self.consolidated_DF.copy().drop(columns = self.parameter_to_consolidate)
                            ilocs = consolidated_DF_copy.index.get_locs([articles_filename])
                            
                            #varrendo as entradas já presentes
                            for line in ilocs:
                                present_vals = consolidated_DF_copy.iloc[line].values
                                #varrendo as novas entradas
                                for new_val in new_vals:
                                    comb_vals = list(present_vals) + [new_val]
                                    entry_dic[(articles_filename, counter)] = comb_vals
                                    counter += 1

                            for index in entry_dic:
                                self.consolidated_DF.loc[ index, : ] = entry_dic[index]
                
                    print('> adicionando na consolidated DF : \n', self.consolidated_DF.loc[ ( articles_filename , ) ,  ] )
                    print(f'Atualizando DataFrame consolidado para {articles_filename}...')
                        
                else:
                    print('\nHá incompatibilidade entre os outputs (função: check_conditions_to_consolidated_DF).')
                    print(f'DataFrame para o {articles_filename} não foi consolidada.')
                    continue

            #salvando a planilha consolidada
            self.consolidated_DF.sort_index(level=[0,1], inplace=True)
            self.consolidated_DF.to_csv(self.diretorio + f'/Outputs/dataframes/_consolidated_DF.csv')    
            self.consolidated_files[self.parameter_to_consolidate] = 'finished'
            save_dic_to_json(self.diretorio + '/Outputs/log/consolidated_files.json', self.consolidated_files)



    def concat_extracted_DFs(self):

        #checando o DF consolidado
        if os.path.exists(self.diretorio + f'/Outputs/log/consolidated_files.json'):
            
            self.consolidated_files = load_dic_from_json(self.diretorio + f'/Outputs/log/consolidated_files.json')

            try:
                self.consolidated_files[self.parameter_to_consolidate]
                
            except KeyError:
                self.consolidated_files[self.parameter_to_consolidate] = 'ongoing'

        else:
            self.consolidated_files = {}
            self.consolidated_files[self.parameter_to_consolidate] = 'ongoing'


        if self.consolidated_files[self.parameter_to_consolidate] != 'finished':
                
            #caso os DFs já tenha sido juntados
            if os.path.exists(self.diretorio + f'/Outputs/dataframes/_concat_DFs_{self.parameter_to_consolidate}.csv'):
                return pd.read_csv(self.diretorio + f'/Outputs/dataframes/_concat_DFs_{self.parameter_to_consolidate}.csv', index_col=[0,1], dtype=object)

            else:    
                #coletando os filenames e o ngrams_to_replace
                inputDFs_to_concat, sec_terms, operantions = get_term_list_from_tuples_strs(self.filenames_to_concatenate)

                #criando uma temp DF
                concat_DF = pd.DataFrame(columns=['Filename', 'Counter', self.parameter_to_consolidate ], dtype=object)
                concat_DF.set_index(['Filename', 'Counter'], inplace=True)

                for i in range(len(inputDFs_to_concat)):

                    #coletando o filename
                    input_DF = inputDFs_to_concat[i]

                    #checando se existe um Data Frame de fragmentos
                    print(f'\nProcurando... {self.diretorio}/Outputs/dataframes/{input_DF}.csv')
                    if os.path.exists(self.diretorio + f'/Outputs/dataframes/{input_DF}.csv'):
                        
                        print(f'Carregando o DataFrame com as intancias extraidas (~/Outputs/dataframes/{input_DF}.csv)')
                        extracted_DF = pd.read_csv(self.diretorio + f'/Outputs/dataframes/{input_DF}.csv', index_col=[0,1], dtype=object)
                        extracted_DF = extracted_DF.sort_index()
                        extracted_paramater = extracted_DF.columns[0]
                        
                        for filename in np.unique(extracted_DF.index.get_level_values(0).values):
                            
                            #checando se já há valores no concat_DF
                            try:
                                existing_values = concat_DF.loc[( filename , ) , self.parameter_to_consolidate].values
                                counter = len(concat_DF.loc[( filename , ) , self.parameter_to_consolidate])
                            except (TypeError, KeyError):
                                existing_values = []
                                counter = 0

                            for val in extracted_DF.loc[ (filename, ) , extracted_paramater ].values:
                                
                                if val not in existing_values:                            
                                    concat_DF.loc[ ( filename , counter ) , self.parameter_to_consolidate ] = val
                                    counter += 1

                    else:
                        print(f' ERRO! Nenhum arquivo encontrado no path: {self.diretorio}/Outputs/dataframes/{input_DF}.csv\n\n')
                        return

                #salvando o DF concatenado
                concat_DF.to_csv(self.diretorio + f'/Outputs/dataframes/_concat_DFs_{self.parameter_to_consolidate}.csv')

                return concat_DF
        
        else:
            print(f'O parâmetro {self.parameter_to_consolidate} já foi consolidado.')
            return None
        


    def check_conditions_to_consolidated_DF(self, filename):
        
        checked = True

        #se já houver algum valor para o filename na DF
        try:        
            self.current_instances_number = len(self.consolidated_DF.index.get_locs([filename]))
        
        #exceção caso o index não exista na DF ou caso o valor de sample_counter seja None
        except KeyError:
            self.current_instances_number = 0

        print('current_instance_number = ', self.current_instances_number)
        print('number_new_instances_to_add = ', self.number_new_instances_extracted)


        #condições para fazer a consolidação do FULL DF
        #para atualização no número de amostras
        cond_match_instances_number = True
        #só se atualiza quando o número de amostra atualizao for igual àquele que está na DF consolidada
        if self.hold_instances_number is True:
            if self.number_new_instances_extracted != self.current_instances_number:
                cond_match_instances_number = False

        #quando o atributo de hold_samples estive ligado, o algoritmo só adiciona parâmetros nas amostras que já tem sample_counter > 0        
        cond_hold_samples = True
        if self.hold_filenames is True:
            #só atualizará se o já houver um número de amostra na DF consolidada
            if (self.current_instances_number == 0):
                cond_hold_samples = False
        else:
            pass                
        
        #caso precise checar as instâncias
        match_previous_instances = True
        if os.path.exists(self.diretorio + f'/Outputs/dataframes/_concat_DFs_{self.match_instances_with_other_parameter}.csv'):
            previous_DF_to_match_instances = pd.read_csv(self.diretorio + f'/Outputs/dataframes/_concat_DFs_{self.match_instances_with_other_parameter}.csv', index_col=[0,1], dtype=object)
            if len(self.consolidated_DF.loc[ (filename , ), self.parameter_to_consolidate ].values) != len(previous_DF_to_match_instances.loc[ (filename , ), self.match_instances_with_other_parameter ].values):
                match_previous_instances = False

        #checando os resultados
        print('hold_instances_number: ', self.hold_instances_number, '; cond_match_instances_number: ', cond_match_instances_number)
        print('hold_filenames: ', self.hold_filenames, '; cond_hold_samples: ', cond_hold_samples)
        print('match_instances_with_other_parameter: ', self.match_instances_with_other_parameter, '; match_previous_instances: ', match_previous_instances)
        #time.sleep(2)

        #se alguma das condições falhou
        if False in (cond_match_instances_number, cond_hold_samples, match_previous_instances):
            checked = False


        return checked