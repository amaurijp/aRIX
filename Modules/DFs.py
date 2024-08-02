#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
import pandas as pd
import numpy as np
import regex as re
import webbrowser as wb

from FUNCTIONS import error_incompatible_strings_input
from FUNCTIONS import get_term_list_from_tuples_strs
from FUNCTIONS import load_dic_from_json
from FUNCTIONS import save_dic_to_json

from functions_PARAMETERS import list_numerical_parameter
from functions_PARAMETERS import list_textual_parameter        
from functions_PARAMETERS import regex_patt_from_parameter
from functions_PARAMETERS import get_physical_units_converted_to_SI

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

        self.input_DF_name = SE_inputs['filename']
        self.output_DF_name = SE_inputs['filename']
        self.file_type = SE_inputs['file_type']
        self.input_parameter = SE_inputs['parameter_to_extract']
        self.extraction_mode = SE_inputs['extraction_mode']

        self.lower_sentence_in_textual_search = SE_inputs['search_inputs']['lower_sentence_for_semantic']
        self.search_token_by_token = SE_inputs['search_inputs']['search_token_by_token']
        self.numbers_extraction_mode = SE_inputs['search_inputs']['numbers_extraction_mode']
        self.filter_unique_results = SE_inputs['search_inputs']['filter_unique_results']


        #abrindo o dic ~/Inputs/ngrams_to_replace
        self.replace_ngrams = False
        self.ngrams_to_replace = load_dic_from_json(self.diretorio + '/Inputs/ngrams_to_replace.json')
        try:
            self.ngrams_to_replace = self.ngrams_to_replace[self.input_parameter]
            self.replace_ngrams = True
            print(f'\nInput ({self.input_parameter}) encontrado no dicionário "~/Inputs/ngrams_to_replace"')
        
        except KeyError:
            print(f'\nInput ({self.input_parameter}) não encontrado no dicionário "~/Inputs/ngrams_to_replace"')



    def extract_textual_parameters(self, text, text_index, parameter, term_list_to_be_found):

        
        extracted_dic = {}
        extracted_dic['text_index'] = text_index
        extracted_dic['parameter'] = parameter
        extracted_dic['all_textual_output'] = []
        
        str_list = []        
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
                        str_list.append(original_token)

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
                    str_list.append(original_ngram)
        
        str_list.sort()
        extracted_dic['all_textual_output'] = str_list
        
        #exportando o número de resultados
        extracted_dic['total_textual_outputs_extracted'] = len(str_list)
        
        #print(extracted_dic)
        #time.sleep(10)
        return extracted_dic



    def extract_numerical_parameters(self, text, text_index, parameter, extract_mode = 'all'):

        #obtendo os padrões regex para encontrar e para não encontrar
        parameters_patterns = regex_patt_from_parameter(parameter)
        
        #print('\nPU_to_find_regex:\n', parameters_patterns['PU_to_find_regex'])
    
        extracted_dic = {}
        extracted_dic['total_num_outputs_extracted'] = 0
        extracted_dic['PUs_extracted'] = None
        extracted_dic['text_index'] = text_index
        extracted_dic['parameter'] = parameter
        extracted_dic['extract_error'] = False
        extracted_dic['all_num_output'] = []
        
        #encontrandos os parâmetros numéricos
        counter = 1
        if re.finditer(parameters_patterns['PU_to_find_regex'], text ):
            for match in re.finditer(parameters_patterns['PU_to_find_regex'], text ):

                PU_found = match.groups()[-2]

                extracted_dic[counter] = {}
                extracted_dic[counter]['nums_not_coverted'] = []
                extracted_dic[counter]['PU_not_converted'] = PU_found.split()

                collected_nums = []
                for find in match.groups():
                    
                    #o padrão regex é para pegar as entradas que só tenham números (sem outros caracteres)
                    if (find is not None) and (find not in collected_nums):

                        #checando se o parâmetro é numérico ou a unidade PU
                        try:
                            rfind = re.sub(r'\s', '', find)
                            float(rfind)
                            extracted_dic[counter]['nums_not_coverted'].append( float(rfind) )
                            extracted_dic['total_num_outputs_extracted'] += 1
                            collected_nums.append( rfind )
                        
                        except ValueError:
                            continue

                counter += 1
        
        #varrendo os findings
        counter = 0
        for key in extracted_dic.keys():
            
            #encontrando as entradas numéricas com os números e as PUs
            try:
                float(key)
                
                if key > 1 and extract_mode == 'one':
                    extracted_dic['extract_error'] = True
                    break

                #fazendo a conversão das unidades
                factor_to_multiply , factor_to_add , PU_in_SI = get_physical_units_converted_to_SI( extracted_dic[key]['PU_not_converted'] )
                
                if None not in (factor_to_multiply , factor_to_add , PU_in_SI):

                    if extracted_dic['PUs_extracted'] is None:
                        extracted_dic['PUs_extracted'] = PU_in_SI
                    
                    extracted_dic[key]['PU'] = PU_in_SI
                    extracted_dic[key]['nums'] = []
                    
                    for num in extracted_dic[key]['nums_not_coverted']:
                        num_converted = round( ( float(num) * factor_to_multiply) + factor_to_add , 9)
                        extracted_dic[key]['nums'].append(num_converted)
                        #nesta lista se coloca todos os números extraidos
                        extracted_dic['all_num_output'].append( num_converted )

                    del extracted_dic[key]['PU_not_converted']
                    del extracted_dic[key]['nums_not_coverted']
                
                else:
                    print('ERRO de conversão das PUs. Checar se as unidades da PU: ', extracted_dic[key]['PU_not_converted'], ' estão na função "get_physical_units_converted_to_SI".')
                    time.sleep(10)


            except ValueError:
                continue

        #print('num_extraction_dic:', extracted_dic)
        #time.sleep(5)
        return extracted_dic



    def generate_search_report(self):
        
        #abrindo o SE report            
        if os.path.exists(self.diretorio + '/Settings/SE_inputs.csv'):
            search_report_DF = pd.read_csv(self.diretorio + '/Settings/SE_inputs.csv', index_col = 0)

            search_report_DF.loc[self.input_DF_name, 'total_extracted'] = self.search_report_dic['export'][self.input_DF_name]['total_finds']
            search_report_DF.loc[self.input_DF_name, 'articles_extracted'] = self.search_report_dic['export'][self.input_DF_name]['article_finds']
            search_report_DF.loc[self.input_DF_name , 'export_status' ] = 'finished'

            search_report_DF.sort_index(inplace=True)
            search_report_DF.to_csv(self.diretorio + '/Settings/SE_inputs.csv')
            print('Salvando o SE report em ~/Settings/SE_inputs.csv')



    def get_data(self):

        print('( Function: get_data )')

        #checando se já existe um output Data Frame para esses parâmetros
        if os.path.exists(self.diretorio + f'/Outputs/dataframes/{self.output_DF_name}.csv'):
            
            self.output_DF = pd.read_csv(self.diretorio + f'/Outputs/dataframes/{self.output_DF_name}.csv', index_col=[0,1], dtype=object)
            self.output_DF.index.names = ['Filename', 'Counter']
            #print(f'Carregando o DataFrame de OUTPUT (~/Outputs/extracted/{self.output_DF_name}.csv)')
        
        else:
            print(f'Output DF {self.output_DF_name}.csv não encontrado.')
            print(f'Criando o output_DF data frame: {self.output_DF_name}.csv')
            
            #caso tenha que ser gerada a output_DF
            self.output_DF = pd.DataFrame(columns=['Filename', 'Counter'], dtype=object)
            self.output_DF.set_index(['Filename', 'Counter'], inplace=True)

        #abrindo o search-extract report
        if os.path.exists(self.diretorio + f'/Outputs/log/se_report.json'):
            
            #carregando o dicionário
            self.search_report_dic = load_dic_from_json(self.diretorio + f'/Outputs/log/se_report.json')
            
            if self.search_report_dic['search'][self.input_DF_name]['searching_status'] != 'finished':
                print(f'Erro! O processo de extração para o search_input {self.input_DF_name} ainda não terminou.' )
                return

            try:
                self.search_report_dic['export']
            except KeyError:
                self.search_report_dic['export'] = {}

            try:
                self.search_report_dic['export'][self.input_DF_name]
            except KeyError:
                self.search_report_dic['export'][self.input_DF_name] = {}
                self.search_report_dic['export'][self.input_DF_name]['last_article_processed'] = None
                self.search_report_dic['export'][self.input_DF_name]['total_finds'] = 0
                self.search_report_dic['export'][self.input_DF_name]['article_finds'] = 0
                
        else:
            print('Erro! LOG counter_se_report não encontrado em ~/outputs/log' )
            print(f'Erro! O processo de extração para o search_input {self.input_DF_name} não foi feito.' )
            return

        #dic para salvar as SI PUs padronizadas para cada features (ou parâmetros)
        if not os.path.exists(self.diretorio + '/Outputs/dataframes/SI_PUs.json'):
            self.SI_PUs_dic_to_record = {}
        else:
            self.SI_PUs_dic_to_record = load_dic_from_json(self.diretorio + '/Outputs/dataframes/SI_PUs.json')

        #coletando os termos a serem encontrados na procura textual
        term_list_to_be_found = get_nGrams_list( [ self.input_parameter ], diretorio=self.diretorio)


        #checando se existe um DF de fragmentos
        print(f'\nProcurando... {self.diretorio}/Outputs/extracted/' + f'{self.input_DF_name}.csv')
        if os.path.exists(self.diretorio + '/Outputs/extracted/' + f'{self.input_DF_name}.csv'):
            
            self.extracted_sents_DF = pd.read_csv(self.diretorio + '/Outputs/extracted/' + f'{self.input_DF_name}.csv', index_col=[0,1], dtype=object)
            print(f'Carregando o DataFrame com os textos extraidos (~/Outputs/extracted/{self.input_DF_name}.csv)')

            #determinando os filenames a varrer
            filenames = list(np.unique(self.extracted_sents_DF.index.get_level_values(0).values))

            try:
                last_article_processed = np.unique(self.output_DF.index.get_level_values(0).values)[-1]
                last_article_index = filenames.index(last_article_processed) + 1

            except IndexError:
                last_article_index = 0

            for filename in filenames[ last_article_index :  ]:
                    
                print(f'\n------------------------------------------')
                print(f'Extracting parameters from {filename}')
                
                #dicionário para coletar os parâmetros numéricos extraídos
                self.parameters_extracted = {}
                self.parameters_extracted['filename'] = filename
                self.parameters_extracted['selected_sent_index'] = None
                self.parameters_extracted['param_type'] = None

                print('\nParameter: ( ', self.input_parameter, ' )') 
                print('Fragments extracted from: ', filename)

                #varrendo as linhas com as sentenças para esse artigo (input_DF)
                for i in self.extracted_sents_DF.loc[ (filename , ) , : ].index:

                    #sentença
                    sent = self.extracted_sents_DF.loc[ (filename , i ) , self.input_DF_name ]
                    
                    #index de sentença
                    sent_index = int( self.extracted_sents_DF.loc[ (filename , i ) , self.input_DF_name + '_index' ] )

                    #coletando a sentença e o sent_index
                    self.parameters_extracted[sent_index] = {}
                    self.parameters_extracted[sent_index]['sent'] = sent
                    self.parameters_extracted[sent_index]['got_parameter_from_sent'] = False
                                                            
                    #Mostrando as sentenças a serem processadas para esse artigo
                    print(f'\nIndex {i} (sent_index {sent_index}):', sent, '\n')
                    
                    #só entramos na sequência abaixo para coletar os parâmetros
                    if self.extraction_mode != 'select_sentences':
                        
                        #checando se o parâmetro irá para a extração numérica
                        if self.input_parameter in list_numerical_parameter():
                            
                            #extraindo os parâmetros numéricos com as unidades físicas
                            numerical_params_extracted_from_sent = self.extract_numerical_parameters(sent, sent_index, self.input_parameter, extract_mode = self.numbers_extraction_mode)
                            
                            #caso tenha sido extraído algum output numérico corretamente
                            if numerical_params_extracted_from_sent['total_num_outputs_extracted'] > 0 and numerical_params_extracted_from_sent['extract_error'] is False:
                                
                                print('> Extracted numerical outputs - n_num_outputs: ', numerical_params_extracted_from_sent['total_num_outputs_extracted'], ' ; SI_units: ', numerical_params_extracted_from_sent['PUs_extracted'])
                                print('>', numerical_params_extracted_from_sent['all_num_output'] )

                                self.parameters_extracted['param_type'] = 'numerical'

                                #adicionando as SI PUs que serão exportadas para a consolidated_DF
                                self.SI_PUs_dic_to_record[self.input_parameter] = numerical_params_extracted_from_sent['PUs_extracted']
                                
                                #caso o método de coleta seja manual
                                if self.extraction_mode == 'collect_parameters_manual':
                                    
                                    while True:
                                        
                                        user_entry = str(input('\nConfirme o(s) valor(es) extraidos (yes/y): '))

                                        if user_entry[0].lower() == 'y':
                                            #coletando os parâmetros extraidos da sentença
                                            self.parameters_extracted[sent_index]['param_captured'] = numerical_params_extracted_from_sent['all_num_output']
                                            self.parameters_extracted[sent_index]['got_parameter_from_sent'] = True
                                            break
                                        
                                        elif user_entry == 'exit':
                                            print('> Abortando função: DataFrames.get_data')
                                            return
                                        
                                        elif user_entry[0].lower() == 'n':
                                            break

                                        else:
                                            print('ERRO > Input inválido (digite "yes", "no" ou "exit")\n')
                                    
                                elif self.extraction_mode == 'collect_parameters_automatic':
                                    #coletando os parâmetros extraidos da sentença [ counter , sent_index , val ]
                                    self.parameters_extracted[sent_index]['param_captured'] = numerical_params_extracted_from_sent['all_num_output']
                                    self.parameters_extracted[sent_index]['got_parameter_from_sent'] = True
                                
                                #time.sleep(2)
                            
                            #caso nenhum output numérico tenha sido exportado
                            else:
                                print(f'> Nenhum parâmetro numérico foi extraído para o parameter: {self.input_parameter}')

                        #checando se o parâmetro irá para a extração textual                                                
                        elif self.input_parameter in list_textual_parameter(diretorio=self.diretorio):

                            textual_params_extracted_from_sent = self.extract_textual_parameters(sent, sent_index, self.input_parameter, term_list_to_be_found)
                            
                            #caso tenha sido extraído algum output textual
                            if textual_params_extracted_from_sent['total_textual_outputs_extracted'] > 0:
                                print('> Extracted textual outputs - n_textual_outputs: ', textual_params_extracted_from_sent['total_textual_outputs_extracted'], ' )')
                                print('> Parâmetros textuais extraídos: ', self.input_parameter)
                                print('> ', textual_params_extracted_from_sent['all_textual_output'] )
                                
                                self.parameters_extracted['param_type'] = 'textual'

                                #caso o método de coleta seja manual
                                if self.extraction_mode == 'collect_parameters_manual':

                                    while True:

                                        user_entry = str(input('\nConfirme o(s) valor(es) extraidos (yes/y): '))

                                        if user_entry.lower() in ('y', 'yes'):
                                            #coletando os parâmetros extraidos da sentença
                                            self.parameters_extracted[sent_index]['param_captured'] = textual_params_extracted_from_sent['all_textual_output']
                                            self.parameters_extracted[sent_index]['got_parameter_from_sent'] = True
                                            break
                                        
                                        elif user_entry == 'exit':
                                            print('> Abortando função: DataFrames.get_data')
                                            return
                                        
                                        elif user_entry.lower() in ('n', 'no'):
                                            break

                                        else:
                                            print('ERRO > Input inválido (digite "yes", "no" ou "exit")\n')
                                
                                elif self.extraction_mode == 'collect_parameters_automatic':
                                    #coletando os parâmetros extraidos da sentença
                                        self.parameters_extracted[sent_index]['param_captured'] = textual_params_extracted_from_sent['all_textual_output']
                                        self.parameters_extracted[sent_index]['got_parameter_from_sent'] = True

                                #time.sleep(2)
                                
                            else:
                                print(f'> Nenhum parâmetro textual foi extraído para o parameter: {self.input_parameter}')
                            
                        #caso o parâmetro introduzido (input_parameter) não esteja na lista do functions_PARAMETERS
                        else:
                            #listar os parâmetros disponíveis para extração
                            available_inputs = list_numerical_parameter() + list_textual_parameter(diretorio=self.diretorio)
                            abort_class = error_incompatible_strings_input('input_parameter', self.input_parameter, available_inputs, class_name = self.class_name)
                            if abort_class is True:
                                return                                        
                                                        
                while True: 
                    try:
                        #escolhendo entre modo de seleção de sentenças ou modo de coleta de parâmetros  
                        if self.extraction_mode == 'select_sentences':
                            print('\nDigite o(s) index(es) das sentenças de interesse (digite valores inteiros)')
                            print('Outros comandos: "+" para ignorar esse artigo e ir para o próximo; "open" para abrir o artigo; e "exit" para sair.')
                            self.param_val = str(input('Index: '))
                            
                        elif self.extraction_mode in ('collect_parameters_automatic', 'collect_parameters_manual'):
                            self.param_val = '*'
                        
                        #processando o input                                        
                        if self.param_val.lower() == 'open':
                            wb.open_new(self.diretorio + '/DB/' + filename + f'.{self.file_type}')
                            continue

                        elif self.param_val.lower() == '*':
                            break
                        
                        elif self.param_val.lower() == '+':
                            break
                        
                        elif self.param_val.lower() == 'exit':
                            print('> Abortando função: DataFrames.get_data')
                            return
                        
                        else:
                            #caso o modo seja para coletar as sentenças para treinamento ML                                                                                
                            if self.extraction_mode == 'select_sentences':
                                try:
                                    selected_sent_key = int(self.param_val)
                                    #verificando se esse parametro inserido é relativo ao valor de key das sentenças coletadas
                                    self.parameters_extracted['selected_sent_index'] = selected_sent_key
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
                if self.param_val != '+':
                                    
                    #caso seja modo de coleta de dados e algum parametro foi extraído
                    if self.extraction_mode != 'select_sentences':
                        
                        #print(self.parameters_extracted)

                        #definindo uma lista para colocar todos os parâmetros coletados
                        all_values_collected_list = []

                        #definindo um dicionário para colocar os outputs modificados
                        self.parameters_extracted[self.input_parameter] = {}
                        self.parameters_extracted[self.input_parameter]['outputs'] = []
                        self.parameters_extracted[self.input_parameter]['extracted_sent_index'] = []
                        #o len_outputs conta quantos valores diferentes foram capturados na extração
                        self.parameters_extracted[self.input_parameter]['len_outputs'] = 0
                        
                        #indicador se algum parametro foi coletado
                        got_any_parameter = False
                            
                        #caso todos os resultados entrem
                        if (self.parameters_extracted['param_type'] == 'numerical'):

                            #varrendo todas as sentenças que tiveram dados extraidos
                            clustered_numbers = ''
                            for sent_index in [ i for i in self.parameters_extracted.keys() if type(i) == int ]:
                                
                                #caso os parametros não foram coletados na sentença (isso acontece somente no modo de coleta manual)
                                if self.parameters_extracted[sent_index]['got_parameter_from_sent'] is False:
                                    continue
                                else:
                                    got_any_parameter = True
                                
                                #varrendo os outputs no formato: [ counter , sent_index , val ]
                                for val in self.parameters_extracted[sent_index]['param_captured']:
                                    
                                    if self.filter_unique_results == True:
                                        if val not in all_values_collected_list:
                                            clustered_numbers += str( round(float( val ), 10) ) + ', '
                                            all_values_collected_list.append(val)
                                        
                                    else:
                                        clustered_numbers += str( round(float( val ), 10) ) + ', '
                                    
                                    self.parameters_extracted[self.input_parameter]['len_outputs'] += 1
                                    
                                    if sent_index not in self.parameters_extracted[self.input_parameter]['extracted_sent_index']:
                                        self.parameters_extracted[self.input_parameter]['extracted_sent_index'].append(sent_index)
                            
                            if len(clustered_numbers) > 0:
                                
                                #tirando o último ', '
                                clustered_numbers = clustered_numbers[ : -2]
                                self.parameters_extracted[self.input_parameter]['outputs'].append( clustered_numbers )
                                
                        #caso todos os resultados textuais entrem
                        elif (self.parameters_extracted['param_type'] == 'textual'):

                            #varrendo todas as sentenças que tiveram dados extraidos
                            for sent_index in [ i for i in self.parameters_extracted.keys() if type(i) == int ]:

                                #caso os parametros não foram coletados na sentença (isso acontece somente no modo de coleta manual)
                                if self.parameters_extracted[sent_index]['got_parameter_from_sent'] is False:
                                    continue
                                else:
                                    got_any_parameter = True

                                #varrendo os outputs no formato: [ counter , sent_index , val ]
                                for val in self.parameters_extracted[sent_index]['param_captured']:
                                    
                                    #caso os parâmetros sejam extraidos somente um vez
                                    if self.filter_unique_results == True:
                                        if val not in self.parameters_extracted[self.input_parameter]['outputs']:
                                            
                                            self.parameters_extracted[self.input_parameter]['outputs'].append( val )
                                            self.parameters_extracted[self.input_parameter]['len_outputs'] += 1
                                    
                                            if sent_index not in self.parameters_extracted[self.input_parameter]['extracted_sent_index']:
                                                self.parameters_extracted[self.input_parameter]['extracted_sent_index'].append(sent_index)
                                    
                                    #caso todos os parâmetros textuais de todas as sentenças sejam extraídos
                                    else:
                                        self.parameters_extracted[self.input_parameter]['outputs'].append( val )
                                        self.parameters_extracted[self.input_parameter]['len_outputs'] += 1
                                        
                                        if sent_index not in self.parameters_extracted[self.input_parameter]['extracted_sent_index']:
                                            self.parameters_extracted[self.input_parameter]['extracted_sent_index'].append(sent_index)
                            
                        #apagando as keys de cada sentença
                        for sent_index in [ i for i in self.parameters_extracted.keys() if type(i) == int ]:
                            del(self.parameters_extracted[sent_index])

                        #gerando a DF com os dados colletados
                        print('\nSummary: parameters extracted from file: ', filename)
                        print(self.parameters_extracted)
                        #time.sleep(5)
                        
                        if got_any_parameter is True:
                            self.convert_parameters_extracted_to_DF()


            #consolidando o report na DF caso seja o ultimo arquivo de procura        
            if filename == self.extracted_sents_DF.index.levels[0][-1]:
                self.generate_search_report()


        else:
            print('Erro! Não foi encontrado um DF de fragamentos de artigos.')
            print('> Abortando a classe: DataFrames')
            return



    def convert_parameters_extracted_to_DF(self):

        if not os.path.exists(self.diretorio + '/Outputs/dataframes'):
            os.makedirs(self.diretorio + '/Outputs/dataframes')

        #número de outputs
        self.instances_extracted = len( self.parameters_extracted[self.input_parameter]['outputs'] )
 
        #gerando e salvando as DF com as sentenças
        if self.instances_extracted >= 1:
            self.export_to_DF()



    def export_to_DF(self):

        print('\nExporting to DF...')
        
        filename = self.parameters_extracted['filename']

        #trabalhando no output_DF
        for i in range( self.instances_extracted ):
            
            #salvando no output_DF
            output_val = self.parameters_extracted[self.input_parameter]['outputs'][i]
            
            self.output_DF.loc[ ( filename , i ) , self.input_parameter ] = output_val
            self.output_DF.loc[ ( filename , i ) , self.input_parameter + '_index' ] = str( self.parameters_extracted[self.input_parameter]['extracted_sent_index'] )
        
        #salvando a output_DF
        self.output_DF.to_csv(self.diretorio + f'/Outputs/dataframes/{self.output_DF_name}.csv')
        print(f'\nSalvando output_DF para {filename}...')

        #contadores para exportar nas DFs de report
        self.search_report_dic['export'][self.input_DF_name]['last_article_processed'] = self.parameters_extracted['filename']
        self.search_report_dic['export'][self.input_DF_name]['total_finds'] += self.instances_extracted
        self.search_report_dic['export'][self.input_DF_name]['article_finds'] += 1
        
        #salvando o DF report
        save_dic_to_json(self.diretorio + f'/Outputs/log/se_report.json', self.search_report_dic)



    def set_settings_to_concatenate(self, dic_inputs = None):
        
        self.parameter_to_consolidate = dic_inputs['parameter']
        self.filenames_to_concatenate = dic_inputs['filenames_to_concatenate']
        self.hold_filenames = dic_inputs['hold_filenames']
        self.hold_instances_number = dic_inputs['hold_instances_number']



    def consolidate_DF(self):

        print('( Function: consolidate_DF )')

        #concatendo os DFs
        concat_extracted_DFs = self.concat_extracted_DFs()

        if self.consolidated_files[self.parameter_to_consolidate] != 'finished':

            #filenames para varrer
            filenames = list(np.unique(concat_extracted_DFs.index.get_level_values(0).values))
            
            for filename in filenames:
                    
                print(f'\n------------------------------------------')
                print(f'Extracting parameters from {filename}')

                self.number_new_instances_extracted = len(concat_extracted_DFs.loc[ (filename , ), self.parameter_to_consolidate].values)

                #coletando o atual número de instancias para o filename
                if self.check_conditions_to_consolidated_DF(filename) is True:
                    
                    #array com todos os valores que serão adicionados
                    new_vals = concat_extracted_DFs.loc[ ( filename, ), self.parameter_to_consolidate].values

                    #enquanto tiver só uma coluna
                    if len(self.consolidated_DF.columns) == 1:
                        counter = 0
                        for new_val in new_vals:
                            self.consolidated_DF.loc[ ( filename , counter ) , self.parameter_to_consolidate ] = new_val                        
                            counter += 1

                    #caso já existam entradas na consolidated DF                    
                    else:
                        entry_dic = {}

                        #fazendo as combinações dos inputs                    
                        consolidated_DF_copy = self.consolidated_DF.copy().drop(columns = self.parameter_to_consolidate)
                        ilocs = consolidated_DF_copy.index.get_locs([filename])
                        
                        #varrendo as entradas já presentes
                        counter = 0
                        for line in ilocs:
                            present_vals = consolidated_DF_copy.iloc[line].values
                            #varrendo as novas entradas
                            for new_val in new_vals:
                                comb_vals = list(present_vals) + [new_val]
                                entry_dic[(filename, counter)] = comb_vals
                                counter += 1

                        for index in entry_dic:
                            self.consolidated_DF.loc[ index, : ] = entry_dic[index]
                
                    self.consolidated_DF.sort_index(level=[0,1], inplace=True)
                    self.consolidated_DF.to_csv(self.diretorio + f'/Outputs/dataframes/consolidated_DF.csv')
                    print('> adicionando na consolidated DF : \n', self.consolidated_DF.loc[ ( filename , ) ,  ])
                    print(f'Atualizando DataFrame consolidado para {filename}...')
                        
                else:
                    print('\nHá incompatibilidade entre os outputs (função: check_conditions_to_consolidated_DF).')
                    print(f'DataFrame para o {filename} não foi consolidada.')
                    continue
                
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
            
            #coletando os filenames e o ngrams_to_replace
            inputDFs_to_concat, sec_terms, operantions = get_term_list_from_tuples_strs(self.filenames_to_concatenate)

            #criando uma temp DF
            temp_DF = pd.DataFrame(columns=['Filename', 'Counter', self.parameter_to_consolidate ], dtype=object)
            temp_DF.set_index(['Filename', 'Counter'], inplace=True)

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
                        
                        #checando se já há valores no temp_DF
                        try:
                            existing_values = temp_DF.loc[( filename , ) , self.parameter_to_consolidate].values
                            counter = len(temp_DF.loc[( filename , ) , self.parameter_to_consolidate])
                        except (TypeError, KeyError):
                            existing_values = []
                            counter = 0

                        for val in extracted_DF.loc[ (filename, ) , extracted_paramater ].values:
                            
                            if val not in existing_values:                            
                                temp_DF.loc[ ( filename , counter ) , self.parameter_to_consolidate ] = val
                                counter += 1

                else:
                    print(f' ERRO! Nenhum arquivo encontrado no path: {self.diretorio}/Outputs/dataframes/{filename}.csv')

            return temp_DF
    
        else:
            print(f'O parâmetro {self.parameter_to_consolidate} já foi consolidado.')
            return None
        


    def check_conditions_to_consolidated_DF(self, filename):
        
        checked = True

        print(f'\nProcurando... {self.diretorio}/Outputs/dataframes/consolidated_DF.csv')
        if not os.path.exists(self.diretorio + f'/Outputs/dataframes/consolidated_DF.csv'):
            print(f'Criando a DF consolidada... (~/Outputs/dataframes/consolidated_DF.csv)')
            self.consolidated_DF = pd.DataFrame(columns=[self.parameter_to_consolidate], index=[[],[]], dtype=object)
            self.consolidated_DF.index.names = ['Filename', 'Counter']
            self.consolidated_DF.to_csv(self.diretorio + f'/Outputs/dataframes/consolidated_DF.csv')

        #carregando a DF consolidada
        else:
            print(f'Abrindo a consolidated DF: {self.diretorio}/Outputs/dataframes/consolidated_DF.csv')
            self.consolidated_DF = pd.read_csv(self.diretorio + f'/Outputs/dataframes/consolidated_DF.csv', index_col=[0,1], dtype=object)
            #adicionando a coluna com o parametro atual a ser extraido
            print(self.parameter_to_consolidate, self.consolidated_DF.columns)
            if self.parameter_to_consolidate not in self.consolidated_DF.columns:
                self.consolidated_DF[self.parameter_to_consolidate] = np.nan


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
                
        #checando os resultados
        print('cond_match_instances_number: ', cond_match_instances_number)
        print('cond_hold_samples: ', cond_hold_samples)
        #time.sleep(2)

        #se alguma das condições falhou
        if False in (cond_match_instances_number, cond_hold_samples):
            checked = False


        return checked