#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
import warnings
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
from functions_PARAMETERS import list_textual_parameter_for_ngrams_search
from functions_PARAMETERS import list_textual_parameter_for_llm_check
from functions_PARAMETERS import list_textual_parameter_for_llm_search 
from functions_PARAMETERS import regex_patt_from_parameter
from functions_PARAMETERS import get_physical_units_converted_to_SI
from functions_PARAMETERS import get_llm_personality_to_check

from functions_TOKENS import get_nGrams_list
from functions_TOKENS import get_tokens_from_sent


##################################################################
class DataFrames(object):
    
    def __init__(self, diretorio = None):
        
        print('\n( Class: DataFrames )')

        self.diretorio = diretorio
        self.class_name = 'DataFrames'

        #loading llm
        self.llm_model = llm('mistral-large:123b-instruct-2407-q2_K')



    def set_settings_for_se(self, SE_inputs = None):
        
        print('\n\n( Function: set_files )')
        print('Setting Files...')

        self.DF_name = SE_inputs['filename']
        self.file_type = SE_inputs['file_type']
        self.input_parameter = SE_inputs['parameter_to_extract']
        
        self.lower_sentence_in_textual_search = SE_inputs['search_inputs']['lower_sentence_for_semantic']
        self.search_token_by_token = SE_inputs['search_inputs']['search_token_by_token']
        self.filter_unique_results = SE_inputs['search_inputs']['filter_unique_results']

        #abrindo o dic ~/Inputs/ngrams_to_replace
        self.replace_ngrams = False
        self.ngrams_to_replace = load_dic_from_json(self.diretorio + '/Inputs/ngrams_to_replace.json')
        if self.input_parameter in list_textual_parameter_for_ngrams_search(diretorio=self.diretorio):
            try:
                self.ngrams_to_replace = self.ngrams_to_replace[self.input_parameter]
                self.replace_ngrams = True
                print(f'\nInput ({self.input_parameter}) encontrado no dicionário "~/Inputs/ngrams_to_replace"')
            
            except KeyError:
                print(f'\nInput ({self.input_parameter}) não encontrado no dicionário "~/Inputs/ngrams_to_replace"')



    def get_data(self):

        print('\n\n( Function: get_data )')

        #coletando os termos a serem encontrados na procura textual
        term_list_to_be_found = None
        if self.input_parameter in list_textual_parameter_for_ngrams_search(diretorio=self.diretorio):
            term_list_to_be_found = get_nGrams_list( [ self.input_parameter ], diretorio=self.diretorio)

        #caso nao haja no ner_rules usa-se a entrada manual introduzida no SE_inputs.csv
        if term_list_to_be_found is None and self.input_parameter not in list_textual_parameter_for_llm_search():
            term_list_to_be_found, sec_terms, operation = get_term_list_from_tuples_strs(self.input_parameter)

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
                self.parameters_extracted['parameter'] = self.input_parameter
                self.parameters_extracted['total_outputs_extracted'] = 0

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
                    print(f'\nSent index {sent_index}:', sent, '\n')
                    
                    #extraindo os valores da sentença
                    self.extract_vals_from_sent(sent, sent_index, term_list_to_be_found)

                #time.sleep(2)
                self.export_to_DF()


            #consolidando o report na DF caso seja o ultimo arquivo de procura        
            if article_filename == self.extracted_sents_DF.index.levels[0][-1]:
                self.generate_search_report()


        else:
            print('Erro! Não foi encontrado um DF de fragamentos de artigos.')
            print('> Abortando a classe: DataFrames')
            return



    def extract_vals_from_sent(self, sent: str, sent_index: int, term_list_to_be_found: list):

        found_parameter = False

        #checando se o parâmetro irá para a extração numérica
        if self.input_parameter in list_numerical_parameter():
            
            found_parameter = True
            #extraindo os parâmetros numéricos com as unidades físicas
            self.extract_numerical_parameters(sent, sent_index, self.input_parameter)
            
        #checando se o parâmetro irá para a extração textual                                                
        cond1 = self.input_parameter.lower() in list_textual_parameter_for_ngrams_search(diretorio=self.diretorio)
        cond2 = self.input_parameter.lower() in list_textual_parameter_for_llm_search()
        cond3 = type(term_list_to_be_found) == list and len(term_list_to_be_found) > 0
        if True in (cond1, cond2, cond3):
            
            found_parameter = True
            #extraindo os parâmetros textuais
            self.extract_textual_parameters(sent, sent_index, self.input_parameter, term_list_to_be_found)
                  
        print('> Extracted outputs - total n_outputs: ', self.parameters_extracted['total_outputs_extracted'] , ')')
        print('> ', self.parameters_extracted[sent_index]['outputs'] )

        #caso o parâmetro introduzido não esteja nas listas do functions_PARAMETERS (textual e numerical)
        if found_parameter is False:
            
            #listar os parâmetros disponíveis para extração
            available_inputs = list_numerical_parameter() + list_textual_parameter_for_llm_search() + list_textual_parameter_for_ngrams_search(diretorio=self.diretorio)
            abort_class = error_incompatible_strings_input('input_parameter', self.input_parameter, available_inputs, class_name = self.class_name)
            if abort_class is True:
                return



    def extract_textual_parameters(self, text: str, text_index: int, parameter: str, term_list_to_be_found: list):
        
        terms_list = []
        
        if parameter.lower() in list_textual_parameter_for_llm_search():
            #extrator via llm
            findings = self.llm_model.extract_textual_parameter(parameter, text)
            time.sleep(0.1)

            if len(findings) > 0:
                for find in findings:
                    terms_list.append(find)
        
        else:
            #varrendo os termos no output do llm
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
        
        #somando para o total de textual_params extraidos para o artigo
        self.parameters_extracted[text_index]['outputs'] = terms_list
        self.parameters_extracted['total_outputs_extracted'] += len( self.parameters_extracted[text_index]['outputs'] )



    def extract_numerical_parameters(self, text: str, text_index: int, parameter: str):

        #extrator via llm
        llm_output = self.llm_model.extract_num_parameter(parameter, text)
        time.sleep(0.1)
        
        self.parameters_extracted[text_index]['outputs'] = [ llm_output ]

        #caso tenha havido extração
        if len(llm_output) > 0:

            #somando para o total de num_params extraidos para o artigo
            self.parameters_extracted['total_outputs_extracted'] += 1



    def export_to_DF(self):

        if not os.path.exists(self.diretorio + '/Outputs/dataframes'):
            os.makedirs(self.diretorio + '/Outputs/dataframes')


        #checando se já existe um output Data Frame para esses parâmetros
        if os.path.exists(self.diretorio + f'/Outputs/dataframes/{self.DF_name}.csv'):
            
            self.output_DF = pd.read_csv(self.diretorio + f'/Outputs/dataframes/{self.DF_name}.csv', index_col=[0,1], dtype=object)
            self.output_DF.index.names = ['Filename', 'Counter']
            #print(f'Carregando o DataFrame de OUTPUT (~/Outputs/extracted/{self.DF_name}_{param}.csv)')
        
        else:
            print(f'Output DF {self.DF_name}.csv não encontrado.')
            print(f'Criando o output_DF data frame: {self.DF_name}.csv')
            
            #caso tenha que ser gerada a output_DF
            self.output_DF = pd.DataFrame(columns=['Filename', 'Counter'], dtype=object)
            self.output_DF.set_index(['Filename', 'Counter'], inplace=True)
        
        counter = 0 
        unique_outputs = []
        for index in [ j for j in self.parameters_extracted.keys() if type(j) == int ]:
            
            #article_name
            article_name = self.parameters_extracted['filename']

            for output in self.parameters_extracted[index]['outputs']:

                if self.filter_unique_results is True and output in unique_outputs:
                    continue
                
                else:
                    if len(output) > 0:
                        unique_outputs.append(output)
                        self.output_DF.loc[ ( article_name , counter ) , self.input_parameter ] = output
                        self.output_DF.loc[ ( article_name , counter ) , self.input_parameter + '_index' ] = str( index )
                        counter += 1
        
        #salvando o último arquivo processado
        self.search_report_dic['export'][self.DF_name]['last_article_processed'] = self.parameters_extracted['filename']
        
        #salvando a output_DF
        try:
            print('\nChecking added lines to DF...')
            print(self.output_DF.loc[ ( article_name , ) , ])
            print('\nExporting to DF...')
            print('> Line added to DF...')
            print(f'\nSalvando output_DF para {article_name}...')
            self.output_DF.to_csv(self.diretorio + f'/Outputs/dataframes/{self.DF_name}.csv')
                    
            #contadores para exportar nas DFs de report
            
            self.search_report_dic['export'][self.DF_name]['total_finds'] += self.parameters_extracted['total_outputs_extracted']
            self.search_report_dic['export'][self.DF_name]['article_finds'] += 1

            print('\nExtract summary')
            print('> total finds:', self.search_report_dic['export'][self.DF_name]['total_finds'])
            print('> articles finds:', self.search_report_dic['export'][self.DF_name]['article_finds'])
        
            #salvando o DF report
            save_dic_to_json(self.diretorio + f'/Outputs/log/se_report.json', self.search_report_dic)
        
        except (TypeError, KeyError):
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



    def set_settings_to_concatenate(self, dic_inputs = None):
        
        self.parameter_to_consolidate = dic_inputs['parameter']
        self.filenames_to_concatenate = dic_inputs['filenames_to_concatenate']
        self.ngrams_to_replace = dic_inputs['ngrams_to_replace']
        self.hold_filenames = dic_inputs['hold_filenames']
        self.hold_instances_number = dic_inputs['hold_instances_number']
        self.filter_unique_results = dic_inputs['filter_unique_results']
        self.parameter_type = dic_inputs['parameter_type']
        self.match_instances_with_other_parameter = dic_inputs['match_instances_with_other_parameter']
        self.check_parameter_relations = dic_inputs['check_parameter_relations']
        self.parameters_used_to_filter = dic_inputs['parameters_used_to_filter']

        #abrindo o dic para remover ngrams
        self.dic_to_remove = load_dic_from_json(self.diretorio + '/Inputs/ngrams_to_remove.json')
        
        #abrindo o dic_index para as sents
        self.index_dic_sents = load_dic_from_json(self.diretorio + '/Outputs/log/index_batch_sents_filtered.json')


    def consolidate_DF(self, consol_DF_name = 'consolidated_DF'):

        print('\n\n( Function: consolidate_DF )')

        print(f'\nProcurando... {self.diretorio}/Outputs/dataframes/_{consol_DF_name}.csv')
        if not os.path.exists(self.diretorio + f'/Outputs/dataframes/_{consol_DF_name}.csv'):
            print(f'Criando a DF consolidada... (~/Outputs/dataframes/_{consol_DF_name}.csv)')
            self.consolidated_DF = pd.DataFrame(columns=[self.parameter_to_consolidate, self.parameter_to_consolidate + '_index'], index=[[],[]], dtype=object)
            self.consolidated_DF.index.names = ['Filename', 'Counter']
            self.consolidated_DF.to_csv(self.diretorio + f'/Outputs/dataframes/_{consol_DF_name}.csv')

        #carregando a DF consolidada
        else:
            print(f'Abrindo a consolidated DF: {self.diretorio}/Outputs/dataframes/_{consol_DF_name}.csv')
            self.consolidated_DF = pd.read_csv(self.diretorio + f'/Outputs/dataframes/_{consol_DF_name}.csv', index_col=[0,1], dtype=object)
            #adicionando a coluna com o parametro atual a ser extraido
            print(self.parameter_to_consolidate, self.consolidated_DF.columns)
            if self.parameter_to_consolidate not in self.consolidated_DF.columns:
                self.consolidated_DF[self.parameter_to_consolidate] = np.nan
                self.consolidated_DF[self.parameter_to_consolidate + '_index'] = np.nan

        #concatendo os DFs
        concat_extracted_DFs = self.concat_extracted_DFs()

        if self.consolidated_files[self.parameter_to_consolidate] != 'finished':

            #filenames para varrer
            articles_filenames = list(np.unique(concat_extracted_DFs.index.get_level_values(0).values))
            
            for articles_filename in articles_filenames:
                    
                print(f'\n------------------------------------------')
                print(f'Extracting parameters from {articles_filename}')

                #quantos valores novos para concatenar
                self.number_new_instances_extracted = len(concat_extracted_DFs.loc[ (articles_filename , ), self.parameter_to_consolidate ].values)
                
                #coletando o atual número de instancias para o articles_filenames
                if self.check_conditions_to_consolidated_DF(articles_filename) is True:

                    counter = 0
                    #caso o articles_filenames não esteja na DF consolidada
                    if articles_filename not in np.unique(self.consolidated_DF.index.get_level_values(0).values) and self.hold_instances_number is False:
                        for new_val, sent_index in zip(concat_extracted_DFs.loc[ (articles_filename, ) , self.parameter_to_consolidate ].values, 
                                                       concat_extracted_DFs.loc[ (articles_filename, ) , self.parameter_to_consolidate + '_index' ].values ):
                            
                            self.consolidated_DF.loc[ ( articles_filename , counter ) , self.parameter_to_consolidate ] = new_val
                            self.consolidated_DF.loc[ ( articles_filename , counter ) , self.parameter_to_consolidate + '_index' ] = sent_index
                            counter += 1

                    else:

                        #enquanto tiver só uma coluna (+ a coluna de index dela)
                        if len(self.consolidated_DF.columns) == 2:
                            for new_val, sent_index in zip(concat_extracted_DFs.loc[ (articles_filename, ) , self.parameter_to_consolidate ].values, 
                                                        concat_extracted_DFs.loc[ (articles_filename, ) , self.parameter_to_consolidate + '_index' ].values ):
                                
                                self.consolidated_DF.loc[ ( articles_filename , counter ) , self.parameter_to_consolidate ] = new_val
                                self.consolidated_DF.loc[ ( articles_filename , counter ) , self.parameter_to_consolidate + '_index' ] = sent_index
                                counter += 1

                        #caso já existam entradas na consolidated DF                    
                        else:
                            
                            #fazendo as combinações dos inputs
                            entry_dic = {}

                            #encontrando as colunas para retirar
                            consolidated_DF_copy = self.consolidated_DF.copy().drop(columns = [self.parameter_to_consolidate, self.parameter_to_consolidate + '_index' ])

                            #encontrando os valores dos indexes para as entradas presentes da consolidated_DF para um determinado artigo
                            ilocs = consolidated_DF_copy.index.get_locs([articles_filename])
                            
                            #varrendo as entradas já presentes
                            for line in ilocs:
                                
                                present_vals = consolidated_DF_copy.iloc[line].values
                                
                                #varrendo as novas entradas
                                for new_val, sent_index in zip(concat_extracted_DFs.loc[ (articles_filename, ) , self.parameter_to_consolidate ].values, 
                                                               concat_extracted_DFs.loc[ (articles_filename, ) , self.parameter_to_consolidate + '_index' ].values ):
                                    
                                    comb_vals = list(present_vals) + [new_val] + [sent_index]
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
            self.consolidated_DF.to_csv(self.diretorio + f'/Outputs/dataframes/_{consol_DF_name}.csv')    
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
                
                #ignorando os erros de lexing sort fo pandas
                #warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

                #dic para coletar valores por artigo
                vals_article = {}
                vals = []
                vals_sent_index = []
                df_indexes = []

                #coletando os filenames para concatenar
                inputDFs_to_concat, sec_terms, operantions = get_term_list_from_tuples_strs(self.filenames_to_concatenate)

                #abrindo o dic com os ngrams para fazer o replace e coletando os keys para replace
                dic_to_replace = load_dic_from_json(self.diretorio + '/Inputs/ngrams_to_replace.json')
                ngrams_dics_to_replace, sec_terms, operantions = get_term_list_from_tuples_strs(self.ngrams_to_replace)

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
                            
                            #armazenando os valores para cada filename
                            try:
                                counter = len(vals_article[filename])
                            except KeyError:
                                vals_article[filename] = []
                                counter = 0
                            
                            for val, sent_index in zip(extracted_DF.loc[ (filename, ) , extracted_paramater ].values, extracted_DF.loc[ (filename, ) , extracted_paramater + '_index' ].values):

                                #tentando substituir por ngrams padronizados
                                for cat in ngrams_dics_to_replace:
                                    try:
                                        val = dic_to_replace[cat][val]
                                    except KeyError:
                                        pass
                                
                                #caso o filter unique results esteja ligado
                                if self.filter_unique_results is True and val in vals_article[filename]:
                                    continue
                                
                                else:
                                    #coletando nas listas
                                    vals.append(val)
                                    vals_sent_index.append(sent_index)
                                    df_indexes.append(( filename, counter))
                                    
                                    #coletando os valores no dic
                                    vals_article[filename].append(val)
                                    counter += 1

                    else:
                        print(f' ERRO! Nenhum arquivo encontrado no path: {self.diretorio}/Outputs/dataframes/{input_DF}.csv\n\n')
                        return

                #criando e salvando o DF
                indexes = pd.MultiIndex.from_tuples(df_indexes , names=['Filename', 'Counter'])
                concat_DF = pd.DataFrame(zip(vals, vals_sent_index), 
                                         columns = [ self.parameter_to_consolidate, self.parameter_to_consolidate + '_index' ], index=indexes)
                concat_DF.sort_index(inplace=True)
                concat_DF.to_csv(self.diretorio + f'/Outputs/dataframes/_concat_DFs_{self.parameter_to_consolidate}.csv')

                return concat_DF
        
        else:
            print(f'O parâmetro "{self.parameter_to_consolidate}" já foi consolidado.')
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


        #para atualização no número de amostras
        cond_match_instances_number = True
        if self.hold_instances_number is True:
            #só se atualiza quando o número de amostra atualizao for igual àquele que está na DF consolidada
            if self.number_new_instances_extracted != self.current_instances_number:
                cond_match_instances_number = False

        #quando o atributo de hold_samples estive ligado, o algoritmo só adiciona parâmetros nas amostras que já tem sample_counter > 0        
        cond_hold_samples = True
        if self.hold_filenames is True:
            #só atualizará se o já houver um número de amostra na DF consolidada
            if (self.current_instances_number == 0):
                cond_hold_samples = False
        
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


    
    def filter_DF(self, consol_DF_name = 'consolidated_DF'):

        print('\n\n( Function: filter_DF )')

        if self.check_parameter_relations is False:
            print(f'O parâmetro {self.parameter_to_consolidate} não será checado')
        
        elif self.check_parameter_relations is True:

            print(f'\nProcurando... {self.diretorio}/Outputs/dataframes/_{consol_DF_name}.csv')
            if not os.path.exists(self.diretorio + f'/Outputs/dataframes/_{consol_DF_name}.csv'):
                print(f'\nERRO! DF consolidada não encontrada em ~/Outputs/dataframes/_{consol_DF_name}.csv\n')
                time.sleep(5)
                return
            
            #vendo se existe um setup para o parâmetro a ser checado
            if self.parameter_to_consolidate not in list_textual_parameter_for_llm_check():
                print(f"\nERRO! O parâmetro '{self.parameter_to_consolidate}' não foi adicionado aos arquivos 'LLM.py' e 'functions_PARAMETERS.py'\n")
                time.sleep(5)
                return

            #carregando a DF consolidada
            else:
                print(f'Abrindo a consolidated DF: {self.diretorio}/Outputs/dataframes/_{consol_DF_name}.csv')
                self.consolidated_DF = pd.read_csv(self.diretorio + f'/Outputs/dataframes/_{consol_DF_name}.csv', index_col=[0,1], dtype=object)

            #procurando a descrição da dependência entre os parâmetros
            params_in_consol_DF, check_params_already_filtered, params_to_be_filtered_along, llm_personality, regex_to_match_llm_output = get_llm_personality_to_check(self.parameter_to_consolidate)
            
            #checango se todos os parâmetros dependentes estão na consolidated DF
            params_columns = []
            for param_ in params_in_consol_DF:
                params_columns.extend( [param_] + [param_ + '_index'] )
                if param_ not in self.consolidated_DF:
                    print(f'\nERRO! A checagem do parêmetro {self.parameter_to_consolidate} depende do parâmetro {param_}.')
                    print(f'Este parâmetro {param_} não está presente em  ~/Outputs/dataframes/_{consol_DF_name}.csv\n')
                    time.sleep(5)
                    return

            #adicionando os parâmetros incluidos no arquivo .csv para dropar os nan
            other_params_to_consider, sec_terms, operantions = get_term_list_from_tuples_strs(self.parameters_used_to_filter)
            for param_ in other_params_to_consider:
                if param_ not in params_columns:
                    params_columns.extend( [param_] + [param_ + '_index'] )

            #caso o self.parameter_to_consolidate ainda não esteja na lista de parâmetros
            params_columns = params_columns + [self.parameter_to_consolidate] + [self.parameter_to_consolidate + '_index'] if self.parameter_to_consolidate not in params_columns else params_columns
            
            consolidated_DF_copy = self.consolidated_DF[ params_columns ].copy()
            
            #retirando os None da DF
            consolidated_DF_copy = consolidated_DF_copy.replace(to_replace={'None':np.nan})
            consolidated_DF_copy = remove_nan(consolidated_DF_copy)

            #consolidated_DF_copy = remove_duplicates(consolidated_DF_copy, columns_to_consider = params_in_consol_DF + [self.parameter_to_consolidate] )
            consolidated_DF_copy = remove_terms_in_DF(consolidated_DF_copy, 
                                                      dic_to_remove = self.dic_to_remove, 
                                                      cats = params_in_consol_DF + [self.parameter_to_consolidate])

            #coletando os indexes
            indexes_to_scan = consolidated_DF_copy.index.tolist()

            if os.path.exists(self.diretorio + f'/Outputs/log/filter_{consol_DF_name}_report.json'):
                #abrindo o dictionário de resumo da varredura
                filter_consol_DF_report = load_dic_from_json(self.diretorio + f'/Outputs/log/filter_{consol_DF_name}_report.json')

                try:
                    index_to_resume = indexes_to_scan.index( tuple( filter_consol_DF_report[f'last_index_processed_{self.parameter_to_consolidate}'] ) ) + 1

                except KeyError:
                    index_to_resume = 0

            else:
                filter_consol_DF_report = {}
                index_to_resume = 0

            last_article_name_tested = None
            last_params_dic = {}
            counter_to_save = 0
            #varrendo a consolidated DF
            for article_name, counter in indexes_to_scan[ index_to_resume : ]:
                
                print(f'\n------------------------------------------')
                print(f'Checking parameter "{self.parameter_to_consolidate}" for : ', article_name, ' , ' , counter)
                counter_to_save += 1
                
                #checando a mundança de artigo
                if last_article_name_tested == None:
                    last_article_name_tested = article_name
                
                elif article_name != last_article_name_tested and last_article_name_tested != None:
                    last_article_name_tested = article_name
                    last_params_dic = {}

                #caso não tenha a instância no dic de fitro
                try:
                    filter_consol_DF_report['(' + article_name + ',' + str(counter) + ')']
                except KeyError:
                    filter_consol_DF_report['(' + article_name + ',' + str(counter) + ')'] = {}

                #dic para levar para o llm
                extracted_param = {}
                for param in params_in_consol_DF + [self.parameter_to_consolidate]: 
                    
                    extracted_param[param] = {}
                    
                    #coletando o param e o sent_index
                    extracted_param[param]['val'] = str( consolidated_DF_copy.loc[ ( article_name, counter ) , param] )
                    extracted_param[param]['sent_index'] = int( consolidated_DF_copy.loc[ ( article_name, counter ) , param + '_index'] )
                
                #caso tenha que ser checado outro parâmetro que já foi checado
                params_filter_check = True
                for param in check_params_already_filtered:
                    try:
                        if extracted_param[param]['val'] != filter_consol_DF_report['(' + article_name + ',' + str(counter) + ')'][param]:
                            params_filter_check = False                            
                            break
                    except KeyError:
                        params_filter_check = False
                        break

                #caso o check esteja correto
                if params_filter_check is True:
                    
                    #testando se a última instância já foi checada e é igual à essa (duplicates)
                    go_to_llm = False
                    for param in extracted_param:
                        try:
                            if extracted_param[param]['val'] != last_params_dic[param]:
                                go_to_llm = True
                                break
                        except KeyError:
                            go_to_llm = True
                            break

                    #fazendo o check com o llm                    
                    if go_to_llm is True:

                        llm_output = self.llm_model.check_parameters(self.parameter_to_consolidate, input_dic = extracted_param, 
                                                                    personality = llm_personality, index_dic_sents = self.index_dic_sents,
                                                                    diretorio = self.diretorio)
                        
                        #caso seja um parâmetro númerico testar
                        if self.parameter_type.lower() == 'numerical' and re.search(regex_to_match_llm_output, llm_output):
                            
                            #gravando só o output do LLM
                            filter_consol_DF_report['(' + article_name + ',' + str(counter) + ')'][self.parameter_to_consolidate] = llm_output
                            print(f'> Checked with LLM - {param} - val: {llm_output}')
                            
                            #outros parâmetros
                            for param in params_to_be_filtered_along:
                                filter_consol_DF_report['(' + article_name + ',' + str(counter) + ')'][param] = extracted_param[param]['val']
                                print(f'> Checked with LLM - {param} - val: {extracted_param[param]["val"]}')

                            #armazenando estes parametros para testar com o próximo (duplicata)
                            last_params_dic[self.parameter_to_consolidate] = llm_output

                            for param in params_in_consol_DF:
                                last_params_dic[param] = extracted_param[param]['val']

                        #caso seja um parâmetro textual
                        elif self.parameter_type.lower() == 'textual' and re.search(regex_to_match_llm_output, llm_output):
                            
                            for param in params_to_be_filtered_along + [self.parameter_to_consolidate]:
                                filter_consol_DF_report['(' + article_name + ',' + str(counter) + ')'][param] = extracted_param[param]['val']
                                print(f'> Checked with LLM - {param} - val: {extracted_param[param]["val"]}')

                            #armazenando estes parametros para testar com o próximo (duplicata)
                            for param in params_in_consol_DF + [self.parameter_to_consolidate]:
                                last_params_dic[param] = extracted_param[param]['val']
                    
                    #caso seja uma duplicata já checada
                    else:
                        for param in params_to_be_filtered_along + [self.parameter_to_consolidate]:
                            filter_consol_DF_report['(' + article_name + ',' + str(counter) + ')'][param] = extracted_param[param]['val']
                            print(f'> Checked (duplicate) - {param} - val: {extracted_param[param]["val"]}')
                        
                        #armazenando estes parametros para testar com o próximo (duplicata)
                        for param in params_in_consol_DF + [self.parameter_to_consolidate]:
                            last_params_dic[param] = extracted_param[param]['val']


                #salvando o DF report
                filter_consol_DF_report[f'last_index_processed_{self.parameter_to_consolidate}'] = ( article_name, counter)                
                time.sleep(0.1)

                #salvando o dicionário
                if counter_to_save == 200 or (article_name == indexes_to_scan[-1][0] and counter == indexes_to_scan[-1][1]):
                    print(f'> Salvando o dicionário em ~/Outputs/log/filter_{consol_DF_name}_report.json')
                    save_dic_to_json(self.diretorio + f'/Outputs/log/filter_{consol_DF_name}_report.json', filter_consol_DF_report)
                    counter_to_save = 0



##################################################################
class Filtered_DF(object):
    
    def __init__(self, diretorio = None):
        
        print('\n( Class: DataFrames )')

        self.diretorio = diretorio
        self.class_name = 'Filtered_DF'

        #dic para salvar as SI PUs padronizadas para cada features (ou parâmetros)
        if not os.path.exists(self.diretorio + '/Outputs/dataframes/SI_PUs.json'):
            self.SI_PUs_dic_to_record = {}
        else:
            self.SI_PUs_dic_to_record = load_dic_from_json(self.diretorio + '/Outputs/dataframes/SI_PUs.json')



    def export_filtered_param_to_DF(self, consol_DF_name = 'consolidated_DF'):
        
        if os.path.exists(self.diretorio + f'/Outputs/dataframes/_{consol_DF_name}_filtered.csv'):
            print('O DF consolidado e filtrado já foi extraído em: ' + self.diretorio + f'/Outputs/dataframes/_{consol_DF_name}_filtered.csv')

        else:
            #carregando o dicionário principal
            if os.path.exists(self.diretorio + f'/Outputs/log/filter_{consol_DF_name}_report.json'):
                #abrindo o dictionário de resumo da varredura
                filter_consol_DF_report = load_dic_from_json(self.diretorio + f'/Outputs/log/filter_{consol_DF_name}_report.json')
            
            else:
                print('\nERRO!')
                print('Arquivo com o dicionário com os parâmetros filtrados não encontrado em: ', self.diretorio + f'/Outputs/log/filter_{consol_DF_name}_report.json')
                return

            #criando a DF
            self.consol_DF = pd.DataFrame(columns=['Filename', 'Counter'], dtype=object)
            self.consol_DF.set_index(['Filename', 'Counter'], inplace=True)

            #dicionários para coletar os valores
            self.dic_to_convert_to_DF = {}
            counter_to_save = 0
            last_article_processed = ''
            counter_i = 0
            for article_name_i in list(filter_consol_DF_report.keys()):

                #caso não seja as keys de resumo
                if re.search(r'ATC[0-9]+', article_name_i):
                    
                    #encontrando o nome do artigo
                    article_name = re.search(r'ATC[0-9]+', article_name_i).group(0)
                    print('> Processing ', article_name)
                    
                    if last_article_processed != article_name:
                        last_article_processed = article_name
                        counter_i = 0

                    try:
                        self.dic_to_convert_to_DF[article_name]

                    except KeyError:
                        self.dic_to_convert_to_DF[article_name] = {}

                    #um key para cada counter
                    self.dic_to_convert_to_DF[article_name][counter_i] = {}
                    
                    #varrendo os parâmetros
                    for param in filter_consol_DF_report[article_name_i]:

                        #tirando o \s no início da str
                        mod_str = re.sub(r'^\s*', '', filter_consol_DF_report[article_name_i][param])

                        #procurando se é parâmetro numérico
                        if param in list_numerical_parameter():
                            
                            #obtendo os padrões regex para encontrar e para não encontrar
                            parameters_patterns = regex_patt_from_parameter(param)

                            #caso tenha encontrado
                            if len(re.findall(parameters_patterns['PU_to_find_regex'], mod_str)) > 0:

                                nums_list = []
                                #varrendo os finds
                                num_finds = re.findall(parameters_patterns['PU_to_find_regex'], mod_str)
                                for i in range(len(num_finds)):
                                    
                                    num_str = num_finds[i]
                                    
                                    #coleta-se o primeiro elemento do grupo
                                    num_raw = float(num_str[0])
                                    
                                    #o elemento 2 possui pode ser um intervalo ou desvio
                                    if num_str[3] != '' and ( (re.search(r'\s*\-\s*', num_str[2]) is not None ) or ( re.search(r'\s*to\s*', num_str[2]) is not None) ):
                                        
                                        #caso seja intervalo, faz se a média entre os números
                                        num_raw = ( float(num_str[0]) + float(num_str[4]) ) /2

                                    elif num_str[3] != '' and re.search(r'(\s*\+\s*\/\s*\-\s*|\s*±\s*)', num_str[2] ): 
                                        #caso seja um desvio, ignora-se
                                        pass
                                
                                    #o elemento 4 do grupo é a unidade PU
                                    PU_not_converted = num_str[5].split()
                                    
                                    #fazendo a conversão das unidades
                                    factor_to_multiply , factor_to_add , PU_in_SI = get_physical_units_converted_to_SI( PU_not_converted )
                                    
                                    if None not in (factor_to_multiply , factor_to_add , PU_in_SI):

                                        num_ = round( (num_raw * factor_to_multiply) + factor_to_add , 9)
                                        nums_list.append(num_)
                                    
                                    else:
                                        print('ERRO de conversão das PUs. Checar se as unidades da PU: ', PU_not_converted, ' estão na função "get_physical_units_converted_to_SI".')
                                        time.sleep(5)                                
                                    
                                    #adicionando as SI PUs que serão exportadas para a consolidated_DF
                                    try:
                                        self.SI_PUs_dic_to_record[param]
                                    except :
                                        self.SI_PUs_dic_to_record[param] = PU_in_SI

                                #coletando os valores numéricos (em caso de mais de um, se usa a média)
                                self.dic_to_convert_to_DF[article_name][counter_i][param] = np.array(nums_list, dtype=float).mean()

                        else:
                            self.dic_to_convert_to_DF[article_name][counter_i][param] = mod_str

                    counter_i += 1
                    counter_to_save += 1

                    #salvando
                    if counter_to_save == 100 or article_name_i == list(filter_consol_DF_report.keys())[-1]:                        
                        self.convert_dic_to_DF(consol_DF_name = consol_DF_name)
                        #apagando o dic
                        self.dic_to_convert_to_DF = {}
                        counter_to_save = 0
                        #time.sleep(5)

            save_dic_to_json(self.diretorio + '/Outputs/dataframes/SI_PUs.json', self.SI_PUs_dic_to_record)  
            print('Salvando o PUs SI em ~/Outputs/dataframes/SI_PUs.json')



    def convert_dic_to_DF(self, consol_DF_name = ''):
        
        for article_name in self.dic_to_convert_to_DF.keys():
            counter = 0
            for i in self.dic_to_convert_to_DF[article_name].keys():
                for param in self.dic_to_convert_to_DF[article_name][i].keys():
                    
                    #preenchendo a DF
                    self.consol_DF.loc[ ( article_name , counter ) , param ] = self.dic_to_convert_to_DF[article_name][i][param]
                
                counter += 1

        self.consol_DF.to_csv(self.diretorio + f'/Outputs/dataframes/_{consol_DF_name}_filtered.csv')
        print(f'\nSalvando... {self.diretorio}/Outputs/dataframes/_{consol_DF_name}_filtered.csv')



###########################################################
def remove_duplicates(df, columns_to_consider = None):

    len_df_before = len(df.index)
    df = df.reset_index().drop_duplicates(subset=['Filename'] + columns_to_consider).set_index(['Filename', 'Counter'])
    print(f'> eliminando duplicatas: {len_df_before - len(df.index)} instâncias removidas.')
    
    return df



def remove_nan(df):

    #Eliminando os NaN
    len_df_before = len(df.index)
    df = df.dropna(axis = 0, how = 'any')
    print(f'> eliminando NaN: {len_df_before - len(df.index)} instâncias removidas.')

    return df



def remove_terms_in_DF(df, dic_to_remove = None, cats = []):

    print('> removing terms from cats in ~/Inputs/ngrams_to_remove.json...')

    #removendo termos presentes no dic ngrams_to_remove
    for cat in cats:
        if (cat in df.columns) and (cat in dic_to_remove.keys()):
            
            #contando quantas aparições o termo term
            unique, counts = np.unique(df[cat].values.astype(str), return_counts=True)
            dic_counter = dict(zip(unique, counts))
            
            for term in dic_to_remove[cat]:
                try:
                    #replacing
                    df = df.reset_index().set_index(cat).drop(index = term).reset_index().set_index(['Filename', 'Counter'])
                    print(f'  removing term {term} in cat: {cat} ({dic_counter[term]} removals)')
                except KeyError:
                    continue
        
        else:
            print(f'  Erro! cat: {cat} não encontrada ou no DF ou no arquivo /Inputs/ngrams_to_remove.json')
    
    return df